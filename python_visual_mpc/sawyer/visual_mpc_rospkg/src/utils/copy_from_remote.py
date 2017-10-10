from paramiko import SSHClient
from scp import SCPClient
import pdb

def scp_pix_distrib_files(policyparams, agentparams):

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='newton4', username='febert')

    # SCPCLient takes a paramiko transport as its only argument
    scp = SCPClient(ssh.get_transport())

    pdb.set_trace()
    i = 0
    filename = '/gen_distrib_t{}.pkl'.format(i)
    scp.get(policyparams['current_dir'] + filename, policyparams['current_dir'] + filename)

    # scp.put('test.txt', '/home/febert/test2.txt')
    for i in range(1,agentparams['T']):
        filename = '/gen_distrib_t{}.pkl'.format(i)
        scp.get(policyparams['current_dir']+filename, policyparams['current_dir']+filename)

        filename = '/gen_image_t{}.pkl'.format(i)
        scp.get(policyparams['current_dir'] + filename, policyparams['current_dir'] + filename)

    scp.close()