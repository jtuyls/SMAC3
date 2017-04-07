
import math
import itertools
import logging
import typing
import random
import time

import numpy as np

from smac.smbo.acquisition import AbstractAcquisitionFunction
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats

from smac.configspace import ConfigurationSpace, Configuration, convert_configurations_to_array


class SelectConfigurations(object):

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 model: RandomForestWithInstances,
                 acquisition_func: AbstractAcquisitionFunction,
                 acq_optimizer: LocalSearch,
                 rng: np.random.RandomState,
                 constant_pipeline_steps,
                 variable_pipeline_steps):
        self.logger = logging.getLogger("Select Configuration")

        self.config_space = scenario.cs
        self.stats = stats
        self.runhistory = runhistory
        self.model = model
        self.acquisition_func = acquisition_func
        self.acq_optimizer = acq_optimizer
        self.rng = rng

        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps

    def run(self, X, Y,
            incumbent,
            num_configurations_by_random_search_sorted: int = 1000,
            num_configurations_by_local_search: int = None,
            random_leaf_size=1):
        print("Run select configuration: rss: {}, ls: {}".format(num_configurations_by_random_search_sorted,
                                                                 num_configurations_by_local_search))
        """Choose next candidate solution with Bayesian optimization.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        num_configurations_by_random_search_sorted: int
             number of configurations optimized by random search
        num_configurations_by_local_search: int
            number of configurations optimized with local search
            if None, we use min(10, 1 + 0.5 x the number of configurations on exp average in intensify calls)

        Returns
        -------
        list
            List of 2020 suggested configurations to evaluate.
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return [x[1] for x in self._get_next_by_random_search(num_points=1)]

        self.model.train(X, Y)

        if self.runhistory.empty():
            incumbent_value = 0.0
        elif incumbent is None:
            # TODO try to calculate an incumbent from the runhistory!
            incumbent_value = 0.0
        else:
            incumbent_value = self.runhistory.get_cost(incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = \
            self._get_next_by_random_search(
                num_configurations_by_random_search_sorted, _sorted=True)

        if num_configurations_by_local_search is None:
            if self.stats._ema_n_configs_per_intensifiy > 0:
                num_configurations_by_local_search = min(
                    10, math.ceil(0.5 * self.stats._ema_n_configs_per_intensifiy) + 1)
            else:
                num_configurations_by_local_search = 10

        # initiate local search with best configurations from previous runs
        configs_previous_runs = self.runhistory.get_all_configs()
        configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
        num_configs_local_search = min(len(configs_previous_runs_sorted), num_configurations_by_local_search)
        next_configs_by_local_search = \
            self._get_next_by_local_search(
                list(map(lambda x: x[1],
                         configs_previous_runs_sorted[:num_configs_local_search])))

        next_configs_by_acq_value = next_configs_by_random_search_sorted + \
                                    next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s" %
            (str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])))
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        # Remove dummy acquisition function value
        next_configs_by_random_search = [x[1] for x in
                                         self._get_next_by_random_search_batch(
                                             num_points=len(next_configs_by_acq_value),
                                             #num_points=num_configs_local_search + num_configurations_by_random_search_sorted,
                                             leaf_size=random_leaf_size)]

        # challengers = list(itertools.chain(*zip(next_configs_by_acq_value,
        #                                         next_configs_by_random_search)))
        iter_next_configs_by_acq_value = iter(next_configs_by_acq_value)
        iter_next_configs_by_random_search = iter(next_configs_by_random_search)
        challengers = [next(iter_next_configs_by_acq_value) if i % (random_leaf_size + 1) == 0 else next(
            iter_next_configs_by_random_search)
                       for i in range(0, len(next_configs_by_acq_value) + len(next_configs_by_random_search))]
        return challengers

    def _get_next_by_random_search(self, num_points=1000, _sorted=False):
        """Get candidate solutions via local search.

        Parameters
        ----------
        num_points : int, optional (default=10)
            Number of local searches and returned values.

        _sorted : bool, optional (default=True)
            Whether to sort the candidate solutions by acquisition function
            value.

        Returns
        -------
        list : (acquisition value, Candidate solutions)
        """

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]

    #### BATCH RANDOM SEARCH ####
    def _get_next_by_random_search_batch(self, num_points=1000, leaf_size=4, _sorted=False):
        """Get candidate solutions via local search.

        Parameters
        ----------
        num_points : int, optional (default=10)
            Number of local searches and returned values.

        _sorted : bool, optional (default=True)
            Whether to sort the candidate solutions by acquisition function
            value.

        Returns
        -------
        list : (acquisition value, Candidate solutions)
        """
        rand_configs = []
        for i in range(0, num_points):
            start_config = self.config_space.sample_configuration(size=1)
            batch_of_configs = [start_config]
            i = 1
            while i < leaf_size:
                try:
                    next_config_combined = self._get_variant_config(start_config=start_config)
                    batch_of_configs.append(next_config_combined)
                    i += 1
                except ValueError as v:
                    pass
            rand_configs.extend(batch_of_configs)
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (Sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (Batch)'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]

    def _get_variant_config(self, start_config, origin=None):
        next_config = start_config
        i = 0
        while i < 1000:
            try:
                next_config = self._combine_configurations(start_config, self.config_space.sample_configuration())
                next_config.origin = origin
                break
            except ValueError as v:
                i += 1
        # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
        return next_config

    def _combine_configurations(self, start_config, new_config):
        constant_values = self._get_values(start_config.get_dictionary(), self.constant_pipeline_steps)
        new_config_values = {}
        new_config_values.update(constant_values)

        variable_values = self._get_values(new_config.get_dictionary(), self.variable_pipeline_steps)
        new_config_values.update(variable_values)

        return Configuration(configuration_space=self.config_space,
                             values=new_config_values)

    def _get_values(self, config_dict, pipeline_steps):
        value_dict = {}
        for step_name in pipeline_steps:
            for hp_name in config_dict:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    value_dict[hp_name] = config_dict[hp_name]
        return value_dict

    #### LOCAL SEARCH ####
    def _get_next_by_local_search(self, init_points=typing.List[Configuration]):
        """Get candidate solutions via local search.

        In case acquisition function values tie, these will be broken randomly.

        Parameters
        ----------
        init_points : typing.List[Configuration]
            initial starting configurations for local search

        Returns
        -------
        list : (acquisition value, Candidate solutions),
               ordered by their acquisition function value
        """
        configs_acq = []

        # Start N local search from different random start points
        for start_point in init_points:
            configuration, acq_val = self.acq_optimizer.maximize(start_point)

            configuration.origin = 'Local Search'
            configs_acq.append((acq_val[0], configuration))

        # shuffle for random tie-break
        random.shuffle(configs_acq, self.rng.rand)

        # sort according to acq value
        # and return n best configurations
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq

    #### HELPER FUNCTIONS ####
    def _sort_configs_by_acq_value(self, configs):
        """ Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value

        """

        #config_array = convert_configurations_to_array(configs)
        acq_values = self.acquisition_func(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]


    def _get_timebound_for_intensification(self, time_spent):
        """ Calculate time left for intensify from the time spent on
        choosing challengers using the fraction of time intended for
        intensification (which is specified in
        scenario.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.scenario.intensification_percentage
        if (frac_intensify <= 0 or frac_intensify >= 1):
            raise ValueError("The value for intensification_percentage-"
                             "option must lie in (0,1), instead: %.2f" % (frac_intensify))
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug("Total time: %.4f, time spent on choosing next "
                          "configurations: %.4f (%.2f), time left for "
                          "intensification: %.4f (%.2f)" % (total_time,
                                                            time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left



class SelectConfigurationsWithMarginalization(SelectConfigurations):

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 model: RandomForestWithInstances,
                 acquisition_func: AbstractAcquisitionFunction,
                 acq_optimizer: LocalSearch,
                 rng: np.random.RandomState,
                 constant_pipeline_steps,
                 variable_pipeline_steps):
        super(SelectConfigurationsWithMarginalization, self).__init__(scenario=scenario,
                                                                     stats=stats,
                                                                     runhistory=runhistory,
                                                                     model=model,
                                                                     acquisition_func=acquisition_func,
                                                                     acq_optimizer=acq_optimizer,
                                                                     rng=rng,
                                                                     constant_pipeline_steps=constant_pipeline_steps,
                                                                     variable_pipeline_steps=variable_pipeline_steps)

    def run(self, X, Y,
            incumbent,
            num_configurations_by_random_search_sorted: int = 1000,
            num_configurations_by_local_search: int = None,
            random_leaf_size=1):
        print("Run select configuration: rss: {}, ls: {}".format(num_configurations_by_random_search_sorted,
                                                                 num_configurations_by_local_search))
        """Choose next candidate solution with Bayesian optimization.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        num_configurations_by_random_search_sorted: int
             number of configurations optimized by random search
        num_configurations_by_local_search: int
            number of configurations optimized with local search
            if None, we use min(10, 1 + 0.5 x the number of configurations on exp average in intensify calls)

        Returns
        -------
        list
            List of 2020 suggested configurations to evaluate.
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return [x[1] for x in self._get_next_by_random_search(num_points=1)]

        self.model.train(X, Y)

        if self.runhistory.empty():
            incumbent_value = 0.0
        elif incumbent is None:
            # TODO try to calculate an incumbent from the runhistory!
            incumbent_value = 0.0
        else:
            incumbent_value = self.runhistory.get_cost(incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        if num_configurations_by_local_search is None:
            if self.stats._ema_n_configs_per_intensifiy > 0:
                num_configurations_by_local_search = min(
                    10, math.ceil(0.5 * self.stats._ema_n_configs_per_intensifiy) + 1)
            else:
                num_configurations_by_local_search = 10

        configs_by_marginalization = self._compute_configs_by_marginalization(
            num_marginalized_configurations_by_random_search=10,
            num_configs_for_marginalization=100,
            num_configurations_by_random_search_sorted=num_configurations_by_random_search_sorted,
            num_configurations_by_local_search=num_configurations_by_local_search)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = \
            self._get_next_by_random_search(
                num_configurations_by_random_search_sorted, _sorted=True)

        # initiate local search with best configurations from previous runs
        configs_previous_runs = self.runhistory.get_all_configs()
        configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
        num_configs_local_search = min(len(configs_previous_runs_sorted), num_configurations_by_local_search)
        next_configs_by_local_search = \
            self._get_next_by_local_search(
                list(map(lambda x: x[1],
                         configs_previous_runs_sorted[:num_configs_local_search])))

        next_configs_by_acq_value = configs_by_marginalization + \
                                    next_configs_by_random_search_sorted + \
                                    next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s" %
            (str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])))
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        # Remove dummy acquisition function value
        # TODO Sometimes 2*(num_configs_local_search + num_configurations_by_random_search_sorted) != len(next_configs_by_acq_value)
        next_configs_by_random_search = [x[1] for x in
                                         self._get_next_by_random_search_batch(
                                             num_points=len(next_configs_by_acq_value),
                                             #num_points=2*(num_configs_local_search + num_configurations_by_random_search_sorted),
                                             leaf_size=random_leaf_size)]

        # challengers = list(itertools.chain(*zip(next_configs_by_acq_value,
        #                                         next_configs_by_random_search)))
        print("LENGTH: {}, {}".format(len(next_configs_by_acq_value), len(next_configs_by_random_search)))
        print("INVESTIGATE LENGTH: {}, {}".format(len(configs_by_marginalization), len(next_configs_by_random_search_sorted)))
        print("INVESTIGATE LENGTH: {}, {}".format(num_configs_local_search, num_configurations_by_random_search_sorted))
        iter_next_configs_by_acq_value = iter(next_configs_by_acq_value)
        iter_next_configs_by_random_search = iter(next_configs_by_random_search)
        challengers = [next(iter_next_configs_by_acq_value) if i % (random_leaf_size + 1) == 0 else next(
            iter_next_configs_by_random_search)
                       for i in range(0, len(next_configs_by_acq_value) + len(next_configs_by_random_search))]
        return challengers

    def _compute_configs_by_marginalization(self,
                                            num_marginalized_configurations_by_random_search=10,
                                            num_configs_for_marginalization=100,
                                            num_configurations_by_local_search=10,
                                            num_configurations_by_random_search_sorted=100):


        #### Compute preprocessor with highest marginalized EI ####
        print("Start select configuration")
        print(num_marginalized_configurations_by_random_search,
              num_configs_for_marginalization,
              num_configurations_by_local_search,
              num_configurations_by_random_search_sorted)
        start_time = time.time()
        # TODO
        configs_by_random_search_sorted_marginalized = \
            self._get_next_by_random_search(
                num_marginalized_configurations_by_random_search, _sorted=True,
                marginalization=True, num_configs_for_marginalization=num_configs_for_marginalization)
        print("Random search for marginalization: {}".format(time.time() - start_time))

        start_time = time.time()
        # TODO Might be to costly to sort all configs from previous runs
        configs_previous_runs = self.runhistory.get_all_configs()
        configs_previous_runs_sorted = self._sort_configs_by_acq_value(configs_previous_runs)
        num_configs_previous_runs_marginalized = min(len(configs_previous_runs_sorted),
                                                     num_configurations_by_local_search)
        configs_previous_runs_sorted_marginalized = \
            self._sort_configs_by_acq_value_marginalization(list(map(lambda x: x[1],
                                                                configs_previous_runs_sorted[
                                                                    :num_configs_previous_runs_marginalized])),
                                                            num_configs_for_marginalization=num_configs_for_marginalization)

        configs_by_acq_value_marginalized = configs_by_random_search_sorted_marginalized + \
                                            configs_previous_runs_sorted_marginalized
        print("Marginalized previous runs preprocessor: {}".format(time.time() - start_time))

        configs_by_acq_value_marginalized.sort(reverse=True, key=lambda x: x[0])
        best_preprocessor_configuration = configs_by_acq_value_marginalized[0][1]
        print("best_preprocessor_configuration: {}".format(best_preprocessor_configuration))

        # Combine preprocessor with highest EI with random classifiers
        start_time = time.time()
        next_marginalized_configs_by_random_search_sorted = self._sort_configs_by_acq_value(
            [self._get_variant_config(start_config=best_preprocessor_configuration,
                                      origin='Random Search marginalization (Sorted)') \
             for i in range(0, num_configurations_by_random_search_sorted)])
        print("Marginalized random search sorted: {}".format(time.time() - start_time))

        start_time = time.time()
        # initiate local search for marginalized preprocessor with best configurations from previous runs
        configs_previous_runs = self.runhistory.get_all_configs()
        # print("configs_previous_runs: {}".format(configs_previous_runs))
        combined_configs_previous_runs = []
        for config in configs_previous_runs:
            try:
                combined_config = self._combine_configurations(start_config=best_preprocessor_configuration,
                                                               new_config=config)
                combined_configs_previous_runs.append(combined_config)
            except ValueError as v:
                pass
        if combined_configs_previous_runs != []:
            combined_configs_previous_runs_sorted = self._sort_configs_by_acq_value(combined_configs_previous_runs)
            # print("combined configs_previous_runs_sorted: {}".format(combined_configs_previous_runs_sorted))

            num_configs_local_search = min(len(combined_configs_previous_runs_sorted),
                                           num_configurations_by_local_search)
            next_marginalized_configs_by_local_search = \
                self._get_next_by_local_search(
                    list(map(lambda x: x[1],
                             combined_configs_previous_runs_sorted[:num_configs_local_search])))
            for _, config in next_marginalized_configs_by_local_search:
                config.origin = "Local Search marginalized"
            print("Marginalized local search sorted: {}".format(time.time() - start_time))
        else:
            next_marginalized_configs_by_local_search = []
            # print("next by local search: {}".format(next_marginalized_configs_by_local_search))
        return next_marginalized_configs_by_random_search_sorted + next_marginalized_configs_by_local_search

    def _get_next_by_random_search(self, num_points=1000, _sorted=False, marginalization=False, num_configs_for_marginalization=100):
        """Get candidate solutions via local search.

        Parameters
        ----------
        num_points : int, optional (default=10)
            Number of local searches and returned values.

        _sorted : bool, optional (default=True)
            Whether to sort the candidate solutions by acquisition function
            value.

        Returns
        -------
        list : (acquisition value, Candidate solutions)
        """

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]2
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (Sorted)'
            if marginalization:
                return self._sort_configs_by_acq_value_marginalization(rand_configs, num_configs_for_marginalization)
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]

    def _sort_configs_by_acq_value_marginalization(self, configs, num_configs_for_marginalization = 100):
        start_time = time.time()
        evaluation_configs = self.config_space.sample_configuration(size=100)
        acq_values = self.acquisition_func.marginalized_prediction(configs=configs, evaluation_configs=evaluation_configs)
        print("Compute marginalization: {}".format(time.time() - start_time))

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        # lst_acq = [(acq_values[ind][0], configs[ind])
        #        for ind in indices_acq[::-1]]

        return [(acq_values[ind], configs[ind])
                for ind in indices[::-1]]

