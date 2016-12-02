""" This file defines the data logger. """
import logging
try:
   import cPickle as pickle
except:
   import pickle


LOGGER = logging.getLogger(__name__)


class DataLogger(object):
    """
    This class pickles data_files into files and unpickles data_files from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data_files to terminal, GUI text/plots, and/or data_files
          files.
    """
    def __init__(self):
        pass

    def pickle(self, filename, data):
        """ Pickle data_files into file specified by filename. """
        pickle.dump(data, open(filename, 'wb'))

    def unpickle(self, filename):
        """ Unpickle data_files from file specified by filename. """
        try:
            return pickle.load(open(filename, 'rb'))
        except IOError:
            LOGGER.debug('Unpickle error. Cannot find file: %s', filename)
            return None
