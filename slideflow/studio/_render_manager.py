import multiprocessing
import slideflow as sf
from typing import Optional

from ._renderer import Renderer, CapturedException

#----------------------------------------------------------------------------

class AsyncRenderManager:

    """Manager to assist with rendering tile-level model predictions."""

    def __init__(self):
        self._closed        = False
        self._is_async      = False
        self._cur_args      = None
        self._cur_result    = None
        self._cur_stamp     = 0
        self._renderer_obj  = None
        self._args_queue    = None
        self._result_queue  = None
        self._process       = None
        self._model_path    = None
        self._live_updates  = False
        self.tile_px        = None
        self.extract_px     = None
        self._addl_render   = []
        self._set_device()

    def _set_device(self) -> None:
        """Set the device for the renderer."""
        if sf.util.torch_available:
            from slideflow.model import torch_utils
            self.device = torch_utils.get_device()
        else:
            self.device = None

    def close(self) -> None:
        """Close the renderer."""
        self._closed = True
        self._renderer_obj = None
        if self._process is not None:
            self._process.terminate()
        self._process = None
        self._args_queue = None
        self._result_queue = None

    @property
    def is_async(self) -> bool:
        """Return whether the renderer is in asynchronous mode.

        Returns:
            bool: Whether the renderer is in asynchronous mode.

        """
        return self._is_async

    def set_renderer(self, renderer_class: type, **kwargs) -> None:
        """Set the renderer class for the renderer.

        Args:
            renderer_class (type): Renderer class to use.

        """
        assert not self._closed
        if self.is_async:
            self._set_args_async(set_renderer=(renderer_class, kwargs))
        else:
            self._renderer_obj = renderer_class(device=self.device, **kwargs)
            for _renderer in self._addl_render:
                self._renderer_obj.add_renderer(_renderer)

    def close_renderer(self) -> None:
        if self.is_async:
            self._set_args_async(close_renderer=True)
        else:
            self._renderer_obj = None

    def add_to_render_pipeline(self, renderer: Renderer) -> None:
        """Add a renderer to the rendering pipeline.

        Args:
            renderer (Renderer): Renderer to add to the pipeline.
                This renderer will be triggered before the main renderer.

        Raises:
            ValueError: If the renderer is in asynchronous mode.

        """
        if self.is_async:
            raise ValueError("Cannot add to rendering pipeline when in "
                             "asynchronous mode.")
        self._addl_render += [renderer]
        if self._renderer_obj is not None:
            self._renderer_obj.add_renderer(renderer)

    def remove_from_render_pipeline(self, renderer: Renderer) -> None:
        """Remove a renderer from the rendering pipeline.

        Args:
            renderer (Renderer): Renderer to remove from the pipeline.

        Raises:
            ValueError: If the renderer is in asynchronous mode.

        """
        if self.is_async:
            raise ValueError("Cannot remove rendering pipeline when in "
                             "asynchronous mode.")
        idx = self._addl_render.index(renderer)
        del self._addl_render[idx]
        if self._renderer_obj is not None:
            self._renderer_obj.remove_renderer(renderer)

    def set_async(self, is_async):
        """Set the renderer to synchronous or asynchronous mode.

        Args:
            is_async (bool): Whether to set the renderer to asynchronous mode.

        """
        self._is_async = is_async

    def set_args(self, **args):
        """Set the arguments for the renderer."""
        assert not self._closed
        if args != self._cur_args or self._live_updates:
            if self._is_async:
                self._set_args_async(**args)
            else:
                self._set_args_sync(**args)
            if not self._live_updates:
                self._cur_args = args

    def _set_args_async(self, **args):
        """Set the arguments for the renderer in asynchronous mode."""
        if self._process is None:
            ctx = multiprocessing.get_context('spawn')
            self._args_queue = ctx.Queue()
            self._result_queue = ctx.Queue()
            self._process = ctx.Process(target=self._process_fn,
                                        args=(self._args_queue,
                                              self._result_queue,
                                              self._model_path,
                                              self._live_updates),
                                        daemon=True)
            self._process.start()
        self._args_queue.put([args, self._cur_stamp])

    def _set_args_sync(self, **args):
        """Set the arguments for the renderer in synchronous mode."""
        if self._renderer_obj is None:
            self._renderer_obj = Renderer(device=self.device)
            for _renderer in self._addl_render:
                self._renderer_obj.add_renderer(_renderer)
            self._renderer_obj._model = self._model
            self._renderer_obj._saliency = self._saliency
        self._cur_result = self._renderer_obj.render(**args)

    def get_result(self):
        """Get the result of the renderer.

        Returns:
            EasyDict: The result of the renderer.

        """
        assert not self._closed
        if self._result_queue is not None:
            while self._result_queue.qsize() > 0:
                result, stamp = self._result_queue.get()
                if stamp == self._cur_stamp:
                    self._cur_result = result
        return self._cur_result

    def clear_result(self):
        """Clear the result of the renderer."""
        assert not self._closed
        self._cur_args = None
        self._cur_result = None
        self._cur_stamp += 1

    def load_model(self, model_path: str) -> None:
        """Load a model for the renderer.

        Args:
            model_path (str): Path to the model.

        """
        if self._is_async:
            self._set_args_async(load_model=model_path)
        elif model_path != self._model_path:
            self._model_path = model_path
            if self._renderer_obj is None:
                self._renderer_obj = Renderer(device=self.device)
                for _renderer in self._addl_render:
                    self._renderer_obj.add_renderer(_renderer)
            self._renderer_obj.load_model(model_path, device=self.device)

    def clear_model(self):
        """Clear the model for the renderer."""
        self._model_path = None
        if self._renderer_obj is not None:
            self._renderer_obj._umap_encoders = None
            self._renderer_obj._model = None
            self._renderer_obj._saliency = None

    @property
    def _model(self):
        if self._renderer_obj is not None:
            return self._renderer_obj._model
        else:
            return None

    @property
    def _saliency(self):
        if self._renderer_obj is not None:
            return self._renderer_obj._saliency
        else:
            return None

    @property
    def _umap_encoders(self):
        if self._renderer_obj is not None:
            return self._renderer_obj._umap_encoders
        else:
            return None

    @staticmethod
    def _process_fn(
        args_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        model_path: Optional[str] = None,
        live_updates: bool = False
    ):
        if sf.util.torch_available:
            from slideflow.model import torch_utils
            device = torch_utils.get_device()
        else:
            device = None
        renderer_obj = Renderer(device=device)
        if model_path:
            renderer_obj.load_model(model_path, device=device)
        while True:
            while args_queue.qsize() > 0:
                args, stamp = args_queue.get()
                if 'close_renderer' in args:
                    renderer_obj = Renderer(device=device)
                if 'set_renderer' in args:
                    renderer_class, kwargs = args['set_renderer']
                    renderer_obj = renderer_class(**kwargs)
                if 'load_model' in args:
                    renderer_obj.load_model(args['load_model'], device=device)
                if 'quit' in args:
                    return
            if (live_updates and not result_queue.qsize()):
                result = renderer_obj.render(**args)
                if 'error' in result:
                    result.error = CapturedException(result.error)

                result_queue.put([result, stamp])
