from paramiko import SSHClient
from scp import SCPClient
import pdb

def scp_pix_distrib_files(policyparams, agentparams):

    ssh = SSHClient()
    ssh.load_system_host_keys()
    # ssh.connect(hostname='newton4', username='febert')
    ssh.connect(hostname='deepthought', username='febert')

    # SCPCLient takes a paramiko transport as its only argument
    scp = SCPClient(ssh.get_transport())

    print('scp pred.pkl')
    scp.get(policyparams['current_dir'] + '/verbose/pred.pkl', policyparams['current_dir'] + '/verbose/pred.pkl')
    # for i in range(1,agentparams['T']):
    #     filename = '/verbose/gen_distrib_t{}.pkl'.format(i)
    #     scp.get(policyparams['current_dir']+filename, policyparams['current_dir']+filename)
    #
    #     filename = '/verbose/gen_image_t{}.pkl'.format(i)
    #     scp.get(policyparams['current_dir'] + filename, policyparams['current_dir'] + filename)

    scp.close()