from .point3d import Point3D

def loadPtNote(self):
    ptnotes = {}
    ptnote_filename = self.path+'/ptnote_TH_'+self.filename_solo+'.txt'
    with open(ptnote_filename,'r') as fp:
        for line in fp:
            line = line[:-1]
            items = line.split('\t')
            pt = Point3D(float(items[1]),float(items[2]),float(items[3]))
            ptnote = {}
            ptnote['pos'] = pt
            ptnote['note'] = items[4]
            ptnote['cat'] = items[-1]

            ptnotes[pt.hashpos()] = ptnote

    return ptnotes
