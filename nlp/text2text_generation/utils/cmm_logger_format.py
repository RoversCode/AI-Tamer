
import json
import logging
import traceback
from collections import OrderedDict
from datetime import date, datetime, time
from inspect import istraceback


RESERVED_ATTRS = (
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module",
    "msecs", "message", "msg", "name", "pathname", "process",
    "processName", "relativeCreated", "stack_info", "thread", "threadName")


class JsonEncoder(json.JSONEncoder):
    """较默认的json encoder支持多一些类型的编码。
    """

    def default(self, obj):
        if isinstance(obj, (date, datetime, time)):
            return obj.isoformat()

        elif istraceback(obj):
            return "".join(traceback.format_tb(obj)).strip()

        elif type(obj) == Exception or isinstance(obj, Exception) or type(obj) == type:
            return str(obj)

        try:
            return super(JsonEncoder, self).default(obj)

        except TypeError:
            try:
                return str(obj)

            except Exception:
                return None


class JsonFormatter(logging.Formatter):
    """将日志格式化为json后输出。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        # standard keys
        output = OrderedDict()
        output["time"] = datetime.fromtimestamp(record.created).isoformat()
        output["level"] = record.levelname
        output["logger"] = record.name
        # output["timestamp"] = record.created
        # output["levelno"] = record.levelno
        output["filename"] = record.filename
        output["lineno"] = record.lineno
        output["msg"] = record.getMessage()

        # extract extra
        for k, v in record.__dict__.items():
            if not isinstance(k, str):
                continue
            if k in RESERVED_ATTRS:
                continue
            if k.startswith("_"):
                continue
            output[k] = v

        # exc_info and stack_info
        if record.exc_info:
            output["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            output["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(dict(output), ensure_ascii=False, separators=(",", ":"), cls=JsonEncoder)


def init():
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
