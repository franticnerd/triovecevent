import json

class POI:
    def __init__(self, poi_id, lat, lng, cat, name):
        self.poi_id = poi_id
        self.lat = lat
        self.lng = lng
        self.cat = cat
        self.name = name

    def __str__(self):
        return '\t'.join([self.name, str(self.lat)+','+str(self.lng), self.cat])

class Tweet:
    def load_tweet(self, line):
        self.line = line
        items = line.split('\x01')
        self.id = long(items[0])
        self.uid = long(items[1])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.datetime = items[4]
        self.ts = int(float(items[5]))%(3600*24)
        self.text = items[6]
        self.words = self.text.split(' ')
        self.raw = items[7] 
        if len(items)>8:
            self.poi_id = items[8]
            self.poi_lat = float(items[9]) if items[9] else items[9]
            self.poi_lng = float(items[10]) if items[10] else items[10]
            self.category = items[11]
            self.poi_name = items[12]
        else:
            self.category = ''

    def load_old_ny(self, line):
        items = line.split('\x01')
        self.id = long(items[0])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.ts = int(float(items[5]))
        self.text = items[6]
        self.words = self.text.split(' ')
        self.category = ''

    def load_checkin(self, line):
        items = line.split('\x01')
        self.id = long(items[0])
        self.lat = float(items[2])
        self.lng = float(items[3])
        self.ts = int(float(items[5]))
        self.text = items[7]
        self.words = self.text.split()
        self.category = items[6]

    # load a clean tweet from a mongo database object
    def load_from_mongo(self, d):
        self.id = d['id']
        self.uid = d['uid']
        self.created_at = d['time']
        self.ts = d['timestamp']
        self.lat = d['lat']
        self.lng = d['lng']
        self.text = d['text']
        # self.words = d['words']
        self.words = d['phrases']
