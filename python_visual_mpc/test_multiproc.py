from multiprocessing import Process, Queue
import time

class Worker(Process):

    def __init__(self, queue, id):
        super(Worker, self).__init__()
        self.queue = queue
        self.id = id

    def run(self):
        print('Worker{} started'.format(self.id))
        # do some initialization here
        for data in iter(self.queue.get, None):
            print('{} getting {}'.format(self.id, data))
            time.sleep(3)
            print('{} getting {} done'.format(self.id, data))


def main():
    request_queue = Queue()
    for i in range(4):
        Worker(request_queue, i).start()
    for data in range(4):
        request_queue.put(data)
    # Sentinel objects to allow clean shutdown: 1 per worker.
    for i in range(4):
        request_queue.put(None)


if __name__ == '__main__':
    main()
