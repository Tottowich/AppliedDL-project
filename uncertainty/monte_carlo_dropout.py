from keras.models import Model
from keras.metrics import Metric
import numpy as np
from typing import List, Tuple, Dict, Union, Optional

class MonteCarloDropoutModel(object):
    def __init__(self, model):
        self.f = Model(model.inputs, model.layers[-1].output)

    def predict(self, x, n_iter=5):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x], training=True).numpy())
        result = np.stack(result, axis=0).std(axis=0)
        return result

class MonteCarloUncertaintyMetric(Metric):
    def __init__(self, monte_carlo_model:MonteCarloDropoutModel=None, name='monte_carlo_uncertainty', **kwargs):
        super(MonteCarloUncertaintyMetric, self).__init__(name=name, **kwargs)
        self.monte_carlo_model = monte_carlo_model if monte_carlo_model is not None else MonteCarloDropoutModel(self.model)
        self.validating_unc = self.add_weight(name='validating_unc', initializer='zeros')
    def set_unc(self,unc):
        self.validating_unc.assign(unc)
    def result(self):
        return self.validating_unc
    def reset_sate(self):
        self.validating_unc.assign(0.0)
    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
        
class ModelWithMonteCarlo(Model):
    def __init__(self, model:Model=None, *args, **kwargs):
        if model is not None: # If model is already defined, wrap it in a MonteCarloDropoutModel
            assert isinstance(model, Model), f"model must be a keras.Model not {type(model)}"
            super().__init__(model.inputs, model.outputs, *args, **kwargs)
        else: # If model is not defined, create a new kears.Model
            super().__init__(*args, **kwargs)
        self.monte_carlo_model = MonteCarloDropoutModel(self)
        self.monte_carlo_metric = MonteCarloUncertaintyMetric(self.monte_carlo_model)
        # Add metric to model
        self.metrics.append(self.monte_carlo_metric)
    def test_step(self, data):
        x, _ = data
        pred = self.monte_carlo_model.predict(x)
        uncertainty = np.mean(pred)
        self.monte_carlo_metric.set_unc(uncertainty)
        return super().test_step(data)