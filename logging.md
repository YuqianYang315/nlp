可以通过以下2种方式来配置logging:

1）使用Python代码显式的创建loggers, handlers和formatters并分别调用它们的配置函数；
2）创建一个包含配置信息的dict，然后把它传递个dictConfig()函数；

# 方法一：
```
imort logging 
```

## 创建一个日志器logger并设置其日志级别为DEBUG
```
logger = logging.getLogger('simple_logger')
<!-- logger = logging.getLogger(__name__) -->
logger.setLevel(logging.DEBUG)
```

### console控制台日志其日志级别为DEBUG
```
console = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(console)
```

### 文件日志其日志级别为DEBUG
```
file_log = logging.FileHandler(log_path,encoding='utf-8')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(file_log)
```


# 方法二：
```
import　logging
import logging.config
import yaml

with open('logging.yml', 'r') as f_conf:
    dict_conf = yaml.load(f_conf)
logging.config.dictConfig(dict_conf)
```

yaml文件内容：
```
# logger config
version: 1
disable_existing_loggers: false

formatters:
  simple:
    format: "%(message)s"
  datetime:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

handlers:
  console:
    class: "logging.StreamHandler"
    level: "DEBUG"
    formatter: "simple"
    stream: "ext://sys.stdout"
  info_file_handler:
    class: "logging.handlers.RotatingFileHandler"
    level: "INFO"
    formatter: "datetime"
    filename: "info.log"
    maxBytes: 10485760
    backupCount: 20
    encoding: "utf8"

root:
  level: "INFO"
  handlers:
    - console
    - info_file_handler
```
