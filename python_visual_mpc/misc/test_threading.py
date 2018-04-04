import threading
import time
import logging

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='(%(threadName)-10s) %(message)s',
                        )

    def daemon():
        logging.debug('Starting')
        time.sleep(2)
        logging.debug('Exiting')

    d = threading.Thread(name='daemon', target=daemon)
    d.setDaemon(True)

    def non_daemon():
        logging.debug('Starting')
        logging.debug('Exiting')

    t = threading.Thread(name='non-daemon', target=non_daemon)

    d.start()
    t.start()

    d.join(1)
    print('d.isAlive()', d.isAlive())
    t.join()

if __name__ == '__main__':
    main()