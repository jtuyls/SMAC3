import time
import numpy as np

from smac.configspace import Configuration, convert_configurations_to_array

class PCAquisitionFunctionWrapper(object):

    def __init__(self, acquisition_func, config_space, runhistory, constant_pipeline_steps, variable_pipeline_steps):
        self.acquisition_func = acquisition_func
        self.config_space = config_space
        self.runhistory = runhistory
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps

    def __call__(self, configs, *args):
        # TODO !! EI
        configs_array_ = convert_configurations_to_array(configs)
        return self.acquisition_func(configs_array_)

    def update(self, **kwargs):
        self.acquisition_func.update(**kwargs)

    def marginalized_prediction(self, configs, evaluation_configs=None):
        start_time = time.time()
        evaluation_configs_values = [self._get_values(evaluation_config.get_dictionary(), self.variable_pipeline_steps) \
                                     for evaluation_config in evaluation_configs]
        marg_acq_values = [self.get_marginalized_acquisition_value(config=config, evaluation_configs_values=evaluation_configs_values) for config in configs]
        # print("MARG ACQUISITION VALUES: {}".format(marg_acq_values))
        print("Compute marginalized acquisition values: {}".format(time.time() - start_time))

        return np.array(marg_acq_values, dtype=np.float64)

    #### HELPER FUNCTIONS ####
    def get_marginalized_acquisition_value(self, config, evaluation_configs_values=None, num_points=100):
        start_time = time.time()
        sample_configs = self._combine_configurations_batch(config, evaluation_configs_values) if evaluation_configs_values \
            else [self._get_variant_config(start_config=config) for i in range(0, num_points)]
        print("List construction: {}".format(time.time() - start_time))

        start_time = time.time()
        caching_discounts = self._compute_caching_discounts(sample_configs,
                                                            self.runhistory.get_cached_configurations())
        print("Compute caching discounts: {}".format(time.time() - start_time))

        start_time= time.time()
        configs_array_ = convert_configurations_to_array(sample_configs)
        print("Compute imputed configs: {}".format(time.time() - start_time))

        #acq_values = self.acquisition_func(imputed_configs, caching_discounts)
        start_time = time.time()
        acq_values = self.acquisition_func(configs_array_)
        print("Acquisition function evaluation: {}".format(time.time() - start_time))
        return np.mean(acq_values)

    def _get_variant_config(self, start_config, origin=None):
        next_config = start_config
        i = 0
        while i < 1000:
            try:
                start_time = time.time()
                sample_config = self.config_space.sample_configuration()
                #print("Sample config: {}".format(time.time() - start_time))
                start_time = time.time()
                next_config = self._combine_configurations(start_config, sample_config)
                #print("Combine config: {}, {}".format(time.time() - start_time, i))
                next_config.origin=origin
                break
            except ValueError as v:
                i += 1
        # TODO hack for now to combine preprocessing part of one configuration with classification part of all the others
        return next_config

    def _combine_configurations(self, start_config, complemented_config):
        constant_values = self._get_values(start_config.get_dictionary(), self.constant_pipeline_steps)
        new_config_values = {}
        new_config_values.update(constant_values)

        variable_values = self._get_values(complemented_config.get_dictionary(), self.variable_pipeline_steps)
        new_config_values.update(variable_values)

        return Configuration(configuration_space=self.config_space,
                             values=new_config_values)

    def _combine_configurations_batch(self, start_config, complemented_configs_values):
        constant_values = self._get_values(start_config.get_dictionary(), self.constant_pipeline_steps)
        batch = []
        for complemented_config_values in complemented_configs_values:
            new_config_values = {}
            new_config_values.update(constant_values)

            new_config_values.update(complemented_config_values)

            try:
                #start_time = time.time()
                config_object = Configuration(configuration_space=self.config_space,
                                              values=new_config_values)
                #print("Constructing configuration: {}".format(time.time() - start_time))
                batch.append(config_object)
            except ValueError as v:
                pass
        return batch

    def _get_values(self, config_dict, pipeline_steps):
        value_dict = {}
        for step_name in pipeline_steps:
            for hp_name in config_dict:
                splt_hp_name = hp_name.split(":")
                if splt_hp_name[0] == step_name:
                    value_dict[hp_name] = config_dict[hp_name]
        return value_dict

    def _compute_caching_discounts(self, configs, cached_configs):
        runtime_discounts = []
        for config in configs:
            discount = 0
            for cached_config in cached_configs:
                discount += self._caching_reduction(config, cached_config)
            runtime_discounts.append(discount)
        return runtime_discounts

    def _caching_reduction(self, config, cached_config):
        '''

        Parameters
        ----------
        config:         the new configuration
        cached_config:  the cached configuration

        Returns
        -------
            The runtime discount for this configuration, given the cached configuration if there is one, otherwise 0
        '''
        config._populate_values()
        r = [key for key in cached_config[0].keys() if config[key] != cached_config[0][key]]
        # print("_caching_reduction: {}".format(r))
        if r == []:
            return cached_config[1]
        return 0

class PCAquisitionFunctionWrapperWithCachingReduction(PCAquisitionFunctionWrapper):

    def __init__(self, acquisition_func, config_space, runhistory, constant_pipeline_steps, variable_pipeline_steps):
        self.acquisition_func = acquisition_func
        self.config_space = config_space
        self.runhistory = runhistory
        self.constant_pipeline_steps = constant_pipeline_steps
        self.variable_pipeline_steps = variable_pipeline_steps

    def __call__(self, configs, *args):
        # TODO !! EI
        caching_discounts = self._compute_caching_discounts(configs, self.runhistory.get_cached_configurations())
        configs_array_ = convert_configurations_to_array(configs)
        return self.acquisition_func(configs_array_, caching_discounts)

    def get_marginalized_acquisition_value(self, config, evaluation_configs_values=None, num_points=100):
        start_time = time.time()
        sample_configs = self._combine_configurations_batch(config,
                                                            evaluation_configs_values) if evaluation_configs_values \
            else [self._get_variant_config(start_config=config) for i in range(0, num_points)]
        print("List construction: {}".format(time.time() - start_time))

        start_time = time.time()
        caching_discounts = self._compute_caching_discounts(sample_configs,
                                                            self.runhistory.get_cached_configurations())
        print("Compute caching discounts: {}".format(time.time() - start_time))

        start_time = time.time()
        configs_array_ = convert_configurations_to_array(sample_configs)
        print("Compute imputed configs: {}".format(time.time() - start_time))

        # acq_values = self.acquisition_func(imputed_configs, caching_discounts)
        start_time = time.time()
        acq_values = self.acquisition_func(configs_array_, caching_discounts)
        print("Acquisition function evaluation: {}".format(time.time() - start_time))
        return np.mean(acq_values)