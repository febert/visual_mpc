#!/usr/bin/env python
import smtplib
import rospy
from std_msgs.msg import String
import _thread
import argparse

class Interruption_notifier(object):
    def __init__(self):
        print('init node')
        rospy.init_node('interruption_notifier', anonymous=True)
        self.last_alive = rospy.get_time()

        parser = argparse.ArgumentParser(description='set robot to use')
        parser.add_argument('robot', type=str, help='robot name')
        args = parser.parse_args()
        self.robotname = args.robot

        _thread.start_new(self.spin_thread, ())


    def spin_thread(self):
        while True:
            tdelta = rospy.get_time() - self.last_alive
            if tdelta > 60*20:
                print('sending notification mail...')
                self.send_mail()
                return
            print('tdelta:', tdelta)
            rospy.sleep(3)

    def callback(self, data):
        self.last_alive = rospy.get_time()

    def start_listener(self):
        rospy.Subscriber("still_alive", String, self.callback)
        rospy.spin()

    def send_mail(self, string = None):
        print('sending notification mail...')
        fromaddr = 'berkeley.sawyer@gmail.com'
        toaddrs = 'frederik.david.ebert@gmail.com'
        if string != None:
            msg = string
        else:
            msg = '{} data collection interrupted'.format(self.robotname)

        username = 'berkeley.sawyer@gmail.com'
        password = 'robots!!'
        server = smtplib.SMTP('smtp.gmail.com:587')

        server.ehlo()
        server.starttls()

        server.login(username, password)
        server.sendmail(fromaddr, toaddrs, msg)
        server.quit()


if __name__ == '__main__':

    n = Interruption_notifier()
    n.start_listener()