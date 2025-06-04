import os
from datetime import timedelta

# Flask settings
DEBUG = os.getenv('FLASK_DEBUG', 'True') == 'True'
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', '5001'))

# API settings
RATE_LIMIT = '100 per minute'
CORS_ORIGINS = ['*']  # Allow all origins since this is an open API

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 