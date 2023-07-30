class Feat():
    def __init__(self,snakelist):
        self._snakelist = snakelist
        self.default_feats = ['Branches','Length','Volume']
        self.feat_func = {'Branches': self.NSnakes, 'Length': self.length, 'Volume': self.volume,
                          'Tortuosity':self.tot}
        self.unit = {'Branches': 1, 'Length': 1, 'Volume': 1,
                          'Tortuosity':1}
        self.unit_name = {'Branches': '', 'Length': '', 'Volume': '',
                          'Tortuosity': ''}

    @property
    def NSnakes(self):
        return len(self._snakelist)

    @property
    def length(self):
        length_sum = 0
        for snakei in self._snakelist:
            length_sum += snakei.length
        return length_sum

    @property
    def volume(self):
        volume_sum = 0
        for snakei in self._snakelist:
            volume_sum += snakei.volume
        return volume_sum

    @property
    def tot(self):
        tot_sum = 0
        if self.length == 0:
            # print('self loop')
            return 1
        for snakei in self._snakelist:
            tot_sum += snakei.tot * snakei.length
        return tot_sum / self.length

    def setUnit(self,res):
        self.unit = {'Branches': 1, 'Length': res, 'Volume': res**3,
                     'Tortuosity': 1}
        self.unit_name = {'Branches': '', 'Length': 'mm', 'Volume': 'mm^3',
                     'Tortuosity': ''}

    def feats(self,feats_names=None,apply_unit=True):
        if feats_names is None:
            feats_names = self.default_feats
        feats_sel = {}
        for feat in feats_names:
            if feat not in self.feat_func:
                print('no such feature',feat)
                continue
            if apply_unit:
                feats_sel[feat] = self.feat_func[feat] * self.unit[feat]
            else:
                feats_sel[feat] = self.feat_func[feat]
        return feats_sel

    def printFeats(self, feats_sel, apply_unit=True, show_title=True):
        print_str = ''
        if show_title:
            if 'pi' in feats_sel:
                print_str += 'Case\t'
            for key in feats_sel:
                if key=='pi':
                    continue
                if apply_unit and self.unit_name[key] != '':
                    print_str += key + ' (' + self.unit_name[key]+ ')\t'
                else:
                    print_str += key + '\t'
            print_str += '\n'
        if 'pi' in feats_sel:
            print_str += feats_sel['pi']+'\t'
        for key in feats_sel:
            if key=='pi':
                continue
            if type(feats_sel[key]) == str:
                print_str += feats_sel[key]+'\t'
            elif type(feats_sel[key]) == int:
                print_str += '%d\t' % feats_sel[key]
            else:
                print_str += '%.1f\t' % feats_sel[key]
        print(print_str)
