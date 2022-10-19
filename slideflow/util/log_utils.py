import logging
import queue
import threading
from tqdm import tqdm


class LogFormatter(logging.Formatter):
    MSG_FORMAT = "%(message)s"
    LEVEL_FORMATS = {
        logging.DEBUG: f"[dim]{MSG_FORMAT}[/]",
        logging.INFO: MSG_FORMAT,
        logging.WARNING: f"[yellow]{MSG_FORMAT}[/]",
        logging.ERROR: f"[red]{MSG_FORMAT}[/]",
        logging.CRITICAL: f"[red bold]{MSG_FORMAT}[/]"
    }

    def format(self, record):
        log_fmt = self.LEVEL_FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    MSG_FORMAT = "%(asctime)s [%(levelname)s] - %(message)s"
    FORMAT_CHARS = ['\033[1m', '\033[2m', '\033[4m', '\033[91m', '\033[92m',
                    '\033[93m', '\033[94m', '\033[38;5;5m', '\033[0m']

    def format(self, record):
        formatter = logging.Formatter(
            fmt=self.MSG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        formatted = formatter.format(record)
        for char in self.FORMAT_CHARS:
            formatted = formatted.replace(char, '')
        return formatted


class MultiProcessingHandler(logging.Handler):
    """Enables logging when using multprocessing.

    From the package `multiprocessing_logging`.
    https://github.com/jruere/multiprocessing-logging
    """
    def __init__(self, name, sub_handler=None):
        super(MultiProcessingHandler, self).__init__()

        if sub_handler is None:
            sub_handler = logging.StreamHandler()
        self.sub_handler = sub_handler

        self.setLevel(self.sub_handler.level)
        self.setFormatter(self.sub_handler.formatter)
        self.filters = self.sub_handler.filters

        self.queue = queue.Queue(-1)
        self._is_closed = False
        # The thread handles receiving records asynchronously.
        self._receive_thread = threading.Thread(target=self._receive, name=name)
        self._receive_thread.daemon = True
        self._receive_thread.start()

    def setFormatter(self, fmt):
        super(MultiProcessingHandler, self).setFormatter(fmt)
        self.sub_handler.setFormatter(fmt)

    def _receive(self):
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break

                record = self.queue.get(timeout=0.2)
                self.sub_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (BrokenPipeError, EOFError):
                break  # The queue was closed by child?
            except queue.Empty:
                pass  # This periodically checks if the logger is closed.
            except:
                from sys import stderr
                from traceback import print_exc

                print_exc(file=stderr)
                raise
        #self.queue.close()
        #self.queue.join_thread()

    def _send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        # ensure that exc_info and args
        # have been stringified. Removes any chance of
        # unpickleable things inside and possibly reduces
        # message size sent over the pipe.
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self._send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._receive_thread.join(5.0)
            self.sub_handler.close()
            super().close()


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flush_line = False

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.flush_line:
                msg = '\r\033[K' + msg
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            print(f"problems with msg {record}")
            self.handleError(record)