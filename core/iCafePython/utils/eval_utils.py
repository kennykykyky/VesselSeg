class MOTMetric():
    def __init__(self):
        self.metric_names = ['TP', 'FN', 'FP', 'IDS', 'MOTA', 'MOTP', 'MT', 'ML',
                             'fragmentation', 'IDTP', 'IDFN', 'IDFP', 'IDF1']
        self.TP_all = 0
        self.FN_all = 0
        self.matched_ids_all = set()
        self.MT_all = 0
        self.ML_all = 0
        self.fragmentation_all = 0
        self.IDS_all = 0
        self.IDTP_all = 0
        self.IDFP_all = 0
        self.IDFN_all = 0
        self.ves_snakelist = None
        self.seg_ves_snakelist = None

    def addSnakeMetric(self,snake_metric):
        TP, FN, matched_ids, MT, ML, fragmentation, IDS, IDTP, IDFN = snake_metric
        self.TP_all += TP
        self.FN_all += FN
        self.matched_ids_all.update(matched_ids)
        self.MT_all += MT
        self.ML_all += ML
        self.fragmentation_all += fragmentation
        self.IDS_all += IDS
        self.IDTP_all += IDTP
        self.IDFN_all += IDFN

    def setSnakelist(self,ves_snakelist,seg_ves_snakelist):
        #ground truth snakelist
        self.ves_snakelist = ves_snakelist
        #predicted snakelist
        self.seg_ves_snakelist = seg_ves_snakelist

    def metrics(self,metrics_sel=None):
        self.T_all = self.ves_snakelist.NPts
        self.FP_all = self.seg_ves_snakelist.NPts - len(self.matched_ids_all)
        self.IDFP_all = self.seg_ves_snakelist.NPts - self.seg_ves_snakelist.link_pts
        self.MOTA = 1 - (self.FN_all + self.FP_all + self.IDS_all) / self.T_all
        self.MOTP = self.ves_snakelist.mean_link_dist
        self.IDF1 = 2*self.IDTP_all/(2*self.IDTP_all+self.IDFP_all+self.IDFN_all)
        metric_dict = {'TP':self.TP_all, 'FN':self.FN_all, 'FP':self.FP_all, 'IDS':self.IDS_all,
                       'MOTA':self.MOTA, 'MOTP':self.MOTP, 'MT':self.MT_all, 'ML':self.ML_all,
                       'fragmentation':self.fragmentation_all, 'IDTP':self.IDTP_all, 'IDFN':self.IDFN_all,
                       'IDFP':self.IDFP_all, 'IDF1':self.IDF1}
        metric_dict_export = {}
        if metrics_sel is None:
            metrics_sel = self.metric_names
        for metric in metrics_sel:
            metric_dict_export[metric] = metric_dict[metric]
        return metric_dict_export

    def printMetrics(self,metrics_sel=None):
        metric_dict = self.metrics(metrics_sel)
        str = ''
        for key in metric_dict:
            str += key+'\t'
        str += '\n'
        for key in metric_dict:
            if type(metric_dict[key]) == int:
                str += '%d\t'%metric_dict[key]
            else:
                str += '%.3f\t'%metric_dict[key]
        print(str)
