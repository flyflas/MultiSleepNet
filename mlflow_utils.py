import os
from contextlib import contextmanager
from numbers import Number


class MLflowTracker:
    """Small optional MLflow wrapper that degrades to no-op when unconfigured."""

    def __init__(self, config):
        self.enabled = False
        self.mlflow = None
        self.tracking_uri = getattr(config, 'mlflow_tracking_uri', '')
        self.experiment_name = getattr(config, 'mlflow_experiment_name', '')

        if not self.tracking_uri:
            print('[INFO] MLflow tracking URI is not configured. MLflow logging disabled.')
            return

        for env_name, attr_name in [
            ('MLFLOW_TRACKING_USERNAME', 'mlflow_tracking_username'),
            ('MLFLOW_TRACKING_PASSWORD', 'mlflow_tracking_password'),
            ('MLFLOW_TRACKING_TOKEN', 'mlflow_tracking_token'),
        ]:
            value = getattr(config, attr_name, '')
            if value:
                os.environ[env_name] = value

        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)

            self.mlflow = mlflow
            self.enabled = True
            print(f'[INFO] MLflow logging enabled: {self.tracking_uri}')
        except Exception as exc:
            print(f'[WARN] MLflow initialization failed. MLflow logging disabled. Reason: {exc}')

    def _safe(self, operation, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            print(f'[WARN] MLflow logging failed and was skipped. Reason: {exc}')
            return None

    @contextmanager
    def start_run(self, run_name=None, nested=False):
        if not self.enabled:
            yield None
            return

        run_context = None
        try:
            run_context = self.mlflow.start_run(run_name=run_name or None, nested=nested)
            run = run_context.__enter__()
        except Exception as exc:
            print(f'[WARN] MLflow start_run failed. MLflow logging disabled. Reason: {exc}')
            self.enabled = False
            yield None
            return

        try:
            yield run
        except BaseException as exc:
            run_context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            try:
                run_context.__exit__(None, None, None)
            except Exception as exc:
                print(f'[WARN] MLflow run close failed. Reason: {exc}')

    def log_params(self, params):
        if not self.enabled:
            return

        clean_params = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean_params[key] = value
            else:
                clean_params[key] = str(value)
        if clean_params:
            self._safe(self.mlflow.log_params, clean_params)

    def log_metrics(self, metrics, step=None):
        if not self.enabled:
            return

        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, Number):
                clean_metrics[key] = float(value)
        if clean_metrics:
            self._safe(self.mlflow.log_metrics, clean_metrics, step=step)

    def log_artifact(self, path, artifact_path=None):
        if not self.enabled:
            return

        if path and os.path.exists(path):
            self._safe(self.mlflow.log_artifact, path, artifact_path=artifact_path)

    def log_artifacts(self, path, artifact_path=None):
        if not self.enabled:
            return

        if path and os.path.isdir(path):
            self._safe(self.mlflow.log_artifacts, path, artifact_path=artifact_path)
