from flask import Flask, render_template, url_for, redirect
import threading
import python_visual_mpc.visual_mpc_core.algorithm.utils as utils
import os
import numpy as np
from python_visual_mpc.video_prediction.misc.makegifs2 import npy_to_mp4

class CEMWebServer:
    def __init__(self, per_iter=3, port=6020, host='0.0.0.0'):
        self._port = port
        self._host = host
        self._per_iter = per_iter
        self._max_page = -1
        self._run_data = []

        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def visualize(self, vd):
        save_dir = vd.agentparams['record'] + '/plan/iter{}_run{}/'.format(vd.t, vd.cem_itr)
        next_page = self._max_page + 1
        static_dir = '{}/static/run{}'.format('/'.join(utils.__file__.split('/')[:-1]), next_page)
        if os.path.exists(static_dir) or os.path.lexists(static_dir):
            os.unlink(static_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        bestindices = vd.scores.argsort()[:vd.K]
        gen_images = (255 * vd.gen_images[bestindices]).astype(np.uint8)
        rows = []

        for i in range(vd.K):
            cols = []
            for n in range(vd.ncam):
                gif_name = 'gen{}_cam{}'.format(i, n)
                npy_to_mp4([gen_images[i, j, n] for j in range(gen_images[i].shape[0])], '{}/{}'.format(save_dir, gif_name))
                cols.append({'url': gif_name, 'caption': 'Gen - {}  cam - {}'.format(i, n)})
            rows.append(cols)

        os.symlink(os.path.abspath(save_dir), os.path.abspath(static_dir))
        self._run_data.append(rows)
        self._max_page = next_page

    def run(self):
        app = Flask(__name__)

        @app.route('/')
        def index():
            return redirect('/run/0')

        @app.route('/run/<int:page_id>')
        def hello_world(page_id):
            page_id = min(int(page_id), self._max_page)

            if page_id < 0:
                return ''' <html> <h1> Still waiting for results! </h1> </html>'''

            run = {'run_id': page_id // self._per_iter, 'iter_id': page_id % self._per_iter}
            run['last_link'] = '/run/{}'.format(max(page_id - 1, 0))
            run['next_link'] = '/run/{}'.format(min(page_id + 1, self._max_page))

            page_data = self._run_data[page_id]
            for rows in page_data:
                for cols in rows:
                    if 'static' not in cols['url']:
                        cols['url'] = url_for('static', filename='run{}/{}.mp4'.format(page_id, cols['url']))

            return render_template('base_template.html', title='MPC Run - {}'.format(page_id), run=run, run_data=page_data)

        @app.after_request
        def add_header(response):
            response.cache_control.max_age = 0
            return response

        app.run(port=self._port, host=self._host)

    @property
    def thread(self):
        return self._thread


if __name__ == '__main__':
    test_server = CEMWebServer()
    test_server.thread.join()