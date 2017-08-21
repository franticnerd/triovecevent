import time
import datetime
from dateutil.parser import parse

class Timestamp:

    def __init__(self, time_string, start_time_string='2000-01-03 00:00:00'):
        self.time_string = time_string
        self.start_time_string = start_time_string

    # calc the timestamp, support either cyclic or absolute
    def gen_timestamp(self, cyclic=None):
        self.calc_abs_timestamp()
        self.calc_cyclic_timestamp(cyclic)

    # calc the absolute timestamp in second
    def calc_abs_timestamp(self):
        start_ts = time.mktime(parse(self.start_time_string).timetuple())
        current_ts = time.mktime(parse(self.time_string).timetuple())
        self.timestamp = current_ts - start_ts

    # get the relative timestamp within one day
    def calc_cyclic_timestamp(self, cyclic):
        if cyclic is None: return
        elif cyclic == 'day': self.timestamp %= (24 * 3600)
        elif cyclic == 'week': self.timestamp %= (7 * 24 * 3600)

    # get the weekday and hour info, weekday is a number from 1 to 7
    def calc_day_hour(self):
        ts = time.mktime(parse(self.time_string).timetuple())
        struct_time = datetime.datetime.fromtimestamp(ts)
        self.weekday = struct_time.isoweekday()
        self.hour = struct_time.hour


    def scale(self, granularity):
        if granularity == 'min': self.timestamp /= 60
        elif granularity == 'hour': self.timestamp /= 3600
        else: print 'Granularity not supported.'


