import numpy as np
import torch
import SimpleITK as sitk
from ..snake import Snake
from ..snakelist import SnakeList
from ..point3d import Point3D
from .ocs_utils import makeOpenA, norm_density
from .cnn_tracker_utils import data_preprocess, get_shell, prob_terminates

class OCSnake(Snake):
    def __init__(self, icafem, snakelist=None, snake=None):
        super(OCSnake, self).__init__()
        self.collision = 0
        self.tail_collision_snake_id = -1
        self.head_collision_snake_id = -1
        self.icafem = icafem
        if 'o' not in self.icafem.I:
            raise ValueError('No img loaded in icafem')
        if 'v' not in  self.icafem.I:
            self.icafem.I['v'] = self.icafem.I['o']
            print('Warning, no vessel img in icafem, use original image instead')

        self.u1, self.sigma1, self.u2, self.sigma2 = icafem.u1, icafem.sigma1, icafem.u2, icafem.sigma2
        self.uv1, self.sigmav1, self.uv2, self.sigmav2 =  icafem.uv1, icafem.sigmav1, icafem.uv2, icafem.sigmav2
        self.head_pt = Point3D(0,0,0)
        self.tail_pt = Point3D(0,0,0)
        if snakelist is None:
            self.snakelist = SnakeList()
            self.snakelist.valid_list = []
        else:
            self.snakelist = snakelist
            if len(self.snakelist.valid_list) != self.snakelist.NSnakes:
                raise ValueError('snakelist sent to ocsnake has no matching valid list')
        self.hit_boundary = False
        if snake is not None:
            self.snake = [swci for swci in snake.snake]
        self.cnn_tracker_model = None
        self.device = None
        self.cnn_tracker_prob_thr = None

    def setCNNTracker(self,cnn_tracker_model, device, cnn_tracker_prob_thr):
        self.cnn_tracker_model = cnn_tracker_model
        self.device = device
        self.cnn_tracker_prob_thr = cnn_tracker_prob_thr
        file_name = self.icafem.getPath('o')
        self.re_spacing_img = sitk.GetArrayFromImage(sitk.ReadImage(file_name))

    def openSnakeStretch_4D(self, alpha, ITER, pt_distance, beta, kappa, gamma,
                                stretchingRatio, collision_dist, minimum_length,
								automatic_merging, max_angle, freeze_body, s_force, snake_id,
								tracing_model, coding_method, sigma_ratio, hit_boundary_dist, vesselnesstracing):
        if self.cnn_tracker_model is not None:
            assert self.device is not None
            assert self.cnn_tracker_prob_thr is not None
            prob_records_head = [0,0,0]
            prob_records_tail = [0,0,0]

        if snake_id==1:
            DEBUG = 0
        else:
            DEBUG=0

        stretchingRatio_head = stretchingRatio
        stretchingRatio_tail = stretchingRatio
        N = self.NP
        m = 8
        eps = 1.192e-7#np.finfo(float).eps
        A = makeOpenA(alpha, beta, N)
        I = np.identity(N)
        invAI = np.linalg.inv (A + I * gamma)
        vnl_Ru = np.array([self.snake[posi].rad for posi in range(N)])

        #evolve 4D snake
        for iter in range(ITER):
            vfx = np.zeros(N)
            vfy = np.zeros(N)
            vfz = np.zeros(N)

            x = np.zeros(N)
            y = np.zeros(N)
            z = np.zeros(N)

            mfx = np.zeros(N)
            mfy = np.zeros(N)
            mfz = np.zeros(N)
            mfr = np.zeros(N)

            for j in range(N):
                x[j] = self.snake[j].pos.x
                y[j] = self.snake[j].pos.y
                z[j] = self.snake[j].pos.z
                if j == 0:
                    v1 = Point3D(self.snake[0].pos.x - self.snake[1].pos.x,
                                 self.snake[0].pos.y - self.snake[1].pos.y,
                                 self.snake[0].pos.z - self.snake[1].pos.z).norm()
                else:
                    v1 = Point3D(self.snake[j].pos.x - self.snake[j-1].pos.x,
                             self.snake[j].pos.y - self.snake[j-1].pos.y,
                             self.snake[j].pos.z - self.snake[j-1].pos.z).norm()

                v2 = Point3D(-1 * v1.x * v1.z,
                             -1 * v1.y * v1.z,
                             pow(v1.x, 2) + pow(v1.y, 2)).norm()
                v3 = Point3D(-1 * v1.y, v1.x, 0).norm()

                force_r_region = 0
                force_region = np.zeros(3)

                #radius
                for k in range(m):
                    theta = (2 * np.pi * k) / m
                    vtemp = Point3D(vnl_Ru[j] * v2.x * np.cos(theta) + vnl_Ru[j] * v3.x * np.sin(theta),
                                    vnl_Ru[j] * v2.y * np.cos(theta) + vnl_Ru[j] * v3.y * np.sin(theta),
                                    vnl_Ru[j] * v2.z * np.cos(theta) + vnl_Ru[j] * v3.z * np.sin(theta))

                    temp_r_pt = Point3D(x[j] + vtemp.x, y[j] + vtemp.y, z[j] + vtemp.z)
                    vtemp = vtemp.norm()
                    oj = np.array([vtemp.x,vtemp.y,vtemp.z])

                    if temp_r_pt.outOfBound(self.icafem.I['o'].shape):
                        continue
                    if vesselnesstracing:
                        Ic = self.icafem.getInt(temp_r_pt, src='v')
                    else:
                        Ic = self.icafem.getInt(temp_r_pt, src='o')

                    ILc = self.icafem.getInt(temp_r_pt, src='l')

                    if ILc != 0:
                        continue

                    if vesselnesstracing:
                        prob1 = norm_density(Ic, self.uv1, self.sigmav1)
                        prob2 = norm_density(Ic, self.uv2, self.sigmav2)
                    else:
                        prob1 = norm_density(Ic, self.u1, self.sigma1)
                        prob2 = norm_density(Ic, self.u2, self.sigma2)

                    force_region += oj * np.log(max(prob1 / max(prob2, eps), eps)) * -1
                    force_r_region += 1 / float(m) * np.log(max(prob1 / max(prob2, eps), eps)) * -1

                mag2 = max(np.sqrt(pow(force_region[0], 2) + pow(force_region[1], 2) + pow(force_region[2], 2) + pow(force_r_region, 2)), eps)
                force_region /= mag2
                force_r_region /= mag2

                mfx[j] = force_region[0]
                mfy[j] = force_region[1]
                mfz[j] = force_region[2]
                mfr[j] = force_r_region

                if 'gvf' in self.icafem.I:
                    gvf = self.icafem.I['gvf']
                    vfx[j] = gvf[0]
                    vfy[j] = gvf[1]
                    vfz[j] = gvf[2]
                else:
                    vfx[j] = 0
                    vfy[j] = 0
                    vfz[j] = 0


            #automate the selection of stretching force
            head_index = self.snake[-1].pos
            tail_index = self.snake[0].pos

            if self.cnn_tracker_model is None:
                stretchingRatio_head, stretchingRatio_tail = self.nextStretchRatio(head_index, tail_index,
                                                                                   vesselnesstracing, stretchingRatio)
                tForce = Point3D(x[0] - x[2], y[0] - y[2], z[0] - z[2]).norm()
                hForce = Point3D(x[-1] - x[-3], y[-1] - y[-3], z[-1] - z[-3]).norm()
            else:
                tForce, curr_r_tail, curr_prob_record_tail = self.CNNTrackerMoveNext('t')
                stretchingRatio_tail = curr_r_tail
                prob_records_tail.pop(0)
                prob_records_tail.append(curr_prob_record_tail)
                prob_mean_tail = sum(prob_records_tail) / len(prob_records_tail)
                if prob_mean_tail > self.cnn_tracker_prob_thr:
                    print('prob_mean_tail above thres',prob_mean_tail)
                    tForce = Point3D(0,0,0)
                #print('tforce',tForce,'rad',curr_r_tail, 'prob',curr_prob_record_tail)

                hForce, curr_r_head, curr_prob_record_head = self.CNNTrackerMoveNext('h')
                stretchingRatio_head = curr_r_head
                prob_records_head.pop(0)
                prob_records_head.append(curr_prob_record_head)
                prob_mean_head = sum(prob_records_head) / len(prob_records_head)
                if prob_mean_head > self.cnn_tracker_prob_thr:
                    print('prob_mean_head above thres',prob_mean_head)
                    hForce = Point3D(0, 0, 0)
                #print('hforce',hForce,'rad',curr_r_head, 'prob',curr_prob_record_head)

            #check for tail leakage and self-intersection
            tail_index_int = tail_index.copy()
            tail_index_int.toIntPos()
            if self.icafem.getInt(tail_index_int,'l') not in [0,snake_id]:
                if DEBUG:
                    print('l 1, tforce 0',self.icafem.getInt(tail_index_int,'l'))
                tForce = Point3D(0,0,0)
                mfx[0],mfy[0],mfz[0],mfr[0] = 0,0,0,0

            leakage = False
            if vesselnesstracing:
                IT = self.icafem.getInt(tail_index, 'v')
                if DEBUG:
                    print('ITv',IT,'ref',self.uv2 + sigma_ratio * self.sigmav2)
                if self.uv2 + sigma_ratio * self.sigmav2 > self.uv1:
                    leakage = norm_density(IT, self.uv1, self.sigmav1) < norm_density(IT, self.uv2, self.sigmav2)
                else:
                    leakage = IT <= self.uv2 + sigma_ratio * self.sigmav2
            else:
                IT = self.icafem.getInt(tail_index, 'o')
                if DEBUG:
                    print('IT', IT, 'ref', self.u2 + sigma_ratio * self.sigma2)
                if self.u2 + sigma_ratio * self.sigma2 > self.u1:
                    leakage = norm_density(IT, self.u1, self.sigma1) < norm_density(IT, self.u2, self.sigma2)
                else:
                    leakage = IT <= self.u2 + sigma_ratio * self.sigma2

            if leakage:
                if DEBUG:
                    print('leak, tforce 0')
                tForce = Point3D(0, 0, 0)
                mfx[0], mfy[0], mfz[0], mfr[0] = 0, 0, 0, 0

            #check for tail collision
            draw_force_tail = 1
            tail_merging = self.checkTailCollision(tail_index, collision_dist, minimum_length, automatic_merging,
                                                   max_angle, snake_id)
            #print('checkTailCollision', tail_index, tail_merging)
            if tail_merging:
                return
            if self.collision == 1 or self.collision == 3:
                x[0] = self.snake[0].pos.x
                y[0] = self.snake[0].pos.y
                z[0] = self.snake[0].pos.z
                tail_index = self.snake[0].pos
                vnl_Ru[0] = self.snake[0].rad
                draw_force_tail = 0

            #check for boundary condition
            boundary_tail = 0
            if tail_index.outOfBox(self.icafem.box,-hit_boundary_dist):
                draw_force_tail = 0
                boundary_tail = 1

            #check for head leakage and self_intersection
            head_index_int = head_index.copy()
            head_index_int.toIntPos()
            if self.icafem.getInt(head_index_int,'l') not in [0,snake_id]:
                if DEBUG:
                    print('hforce 1',self.icafem.getInt(head_index_int,'l'))
                hForce = Point3D(0,0,0)
                mfx[-1] = 0
                mfy[-1] = 0
                mfz[-1] = 0
                mfr[-1] = 0

            leakage = False
            if vesselnesstracing:
                IH = self.icafem.getInt(head_index,'v')
                if DEBUG:
                    print('IHv',IH,'ref',self.uv2 + sigma_ratio * self.sigmav2)
                if self.uv2 + sigma_ratio * self.sigmav2 > self.uv1:
                    leakage = norm_density(IH, self.uv1, self.sigmav1) < norm_density(IH, self.uv2, self.sigmav2)
                else:
                    leakage = IH <= self.uv2 + sigma_ratio * self.sigmav2
            else:
                IH = self.icafem.getInt(head_index, 'o')
                if DEBUG:
                    print('IH', IH, 'ref', self.u2 + sigma_ratio * self.sigma2)
                if self.u2 + sigma_ratio * self.sigma2 > self.u1:
                    leakage = norm_density(IH, self.u1, self.sigma1) < norm_density(IH, self.u2, self.sigma2)
                else:
                    leakage = IH <= self.u2 + sigma_ratio * self.sigma2

            if leakage:
                if DEBUG:
                    print('leak, hforce 0')
                hForce = Point3D(0,0,0)
                mfx[-1] = 0
                mfy[-1] = 0
                mfz[-1] = 0
                mfr[-1] = 0

            draw_force_head = 1

            head_merging = self.checkHeadCollision(head_index, collision_dist, minimum_length, automatic_merging,
                                                max_angle, snake_id)
            #print('checkHeadCollision',head_index,head_merging)

            if head_merging:
                return
            if self.collision == 2 or self.collision == 3:
                x[-1] = self.snake[-1].pos.x
                y[-1] = self.snake[-1].pos.y
                z[-1] = self.snake[-1].pos.z
                head_index = self.snake[-1].pos
                vnl_Ru[-1] = self.snake[-1].rad
                draw_force_head = 0
            #check for boundary condition
            boundary_head = 0
            if head_index.outOfBox(self.icafem.box,-hit_boundary_dist):
                draw_force_head = 0
                boundary_head = 1

            vfx[0] = (vfx[0] + stretchingRatio_head * tForce.x) * draw_force_tail
            vfy[0] = (vfy[0] + stretchingRatio_head * tForce.y) * draw_force_tail
            vfz[0] = (vfz[0] + stretchingRatio_head * tForce.z) * draw_force_tail
            mfx[0] = mfx[0] * draw_force_tail
            mfy[0] = mfy[0] * draw_force_tail
            mfz[0] = mfz[0] * draw_force_tail
            mfr[0] = mfr[0] * draw_force_tail

            vfx[-1] = (vfx[-1] + stretchingRatio_tail * hForce.x) * draw_force_head
            vfy[-1] = (vfy[-1] + stretchingRatio_tail * hForce.y) * draw_force_head
            vfz[-1] = (vfz[-1] + stretchingRatio_tail * hForce.z) * draw_force_head
            mfx[-1] = mfx[-1] * draw_force_head
            mfy[-1] = mfy[-1] * draw_force_head
            mfz[-1] = mfz[-1] * draw_force_head
            mfr[-1] = mfr[-1] * draw_force_head

            r_head = 0
            r_tail = 0
            if self.collision ==1 or self.collision == 3:
                r_tail = vnl_Ru[0]
            elif self.collision == 2 or self.collision == 3:
                r_head = vnl_Ru[-1]
            if DEBUG:
                print('iter',iter,'z',z,'vfz',vfz,'mfz',mfz)
            x = np.dot(invAI, (x * gamma + vfx - mfx))
            y = np.dot(invAI, (y * gamma + vfy - mfy))
            z = np.dot(invAI, (z * gamma + vfz - mfz))
            vnl_Ru = np.dot(invAI, (vnl_Ru * gamma - mfr))

            if self.collision == 1 or self.collision == 3:
                x[0] = tail_index.x
                y[0] = tail_index.y
                z[0] = tail_index.z
                vnl_Ru[0] = r_tail
            elif self.collision == 2 or self.collision == 3:
                x[-1] = head_index.x
                y[-1] = head_index.y
                z[-1] = head_index.z
                vnl_Ru[-1] = r_head
            for k in range(len(x)):
                if freeze_body:
                    if self.collision == 1 and k <N/2:
                        continue
                    if self.collision == 2 and k > N/2:
                        continue
                self.snake[k].pos.x = x[k]
                self.snake[k].pos.y = y[k]
                self.snake[k].pos.z = z[k]
                self.snake[k].pos.bound(self.icafem.SM,self.icafem.SN,self.icafem.SZ)
                self.snake[k].rad = max(vnl_Ru[k],eps)
                self.reset()

            if boundary_head == 1 and boundary_tail == 1:
                self.hit_boundary = True
                break

        #end evolve 4D snake
        #check for NaN
        self.checkNaN()

        #resampling
        resampled_snake = self.resampleSnake(pt_distance)
        self.snake = resampled_snake.snake
        if DEBUG:
            print('Number of Points', self.NP, 'Length', self.length)

    def nextStretchRatio(self,head_index,tail_index, vesselnesstracing, stretchingRatio):
        eps = 1.192e-7#np.finfo(float).eps
        if vesselnesstracing:
            Ic_h = self.icafem.getInt(head_index,'v')
            prob1_h = norm_density(Ic_h, self.uv1, self.sigmav1)
            prob2_h = norm_density(Ic_h, self.uv2, self.sigmav2)
        else:
            Ic_h = self.icafem.getInt(head_index,'o')
            prob1_h = norm_density(Ic_h, self.u1, self.sigma1)
            prob2_h = norm_density(Ic_h, self.u2, self.sigma2)

        stretchingRatio_head = np.log(max(prob1_h / max(prob2_h, eps), eps))
        #print('stretchingRatio_head',stretchingRatio_head)
        if stretchingRatio_head < stretchingRatio:
            stretchingRatio_head = stretchingRatio

        if vesselnesstracing:
            Ic_t = self.icafem.getInt(tail_index,'v')
            prob1_h = norm_density(Ic_t, self.uv1, self.sigmav1)
            prob2_h = norm_density(Ic_t, self.uv2, self.sigmav2)
        else:
            Ic_t = self.icafem.getInt(tail_index,'o')
            prob1_h = norm_density(Ic_t, self.u1, self.sigma1)
            prob2_h = norm_density(Ic_t, self.u2, self.sigma2)

        stretchingRatio_tail = np.log(max(prob1_h / max(prob2_h, eps), eps))
        #print('stretchingRatio_tail',stretchingRatio_tail)

        if stretchingRatio_tail < stretchingRatio:
            stretchingRatio_tail = stretchingRatio
        return stretchingRatio_head, stretchingRatio_tail

    def checkTailCollision(self, index, collision_dist, minimum_length, automatic_merging,
                           max_angle, snake_id):
        DEBUG = 1
        angle_th = max_angle
        merging = False
        #collision 1: tail, 3: both
        if self.collision == 1 or self.collision == 3:
            return merging
        SM = self.icafem.SM
        SN = self.icafem.SN
        SZ = self.icafem.SZ
        overlap = False

        tail_pt = index

        L3 = self.length
        for ix in range(-collision_dist,collision_dist+1):
            for iy in range(-collision_dist, collision_dist + 1):
                for iz in range(-collision_dist, collision_dist + 1):
                    temp_pt = Point3D(index.x+ix,index.y+iy,index.z+iz)
                    temp_pt.bound(SM,SN,SZ)
                    new_index = temp_pt.copy()
                    new_index.toIntPos()
                    id = int(self.icafem.getInt(new_index,'l'))
                    if id!=0 and id!=snake_id:
                        #check if snake is removed
                        if self.snakelist.valid_list[id-1]==False:
                            continue
                        overlap = True
                        #find nearest point at the traced snake
                        dist_temp = [tail_pt.dist(self.snakelist[id-1][i].pos) for i in range(self.snakelist[id-1].NP)]
                        pt_id = np.argmin(dist_temp)
                        if automatic_merging:
                            L1 = self.snakelist[id - 1].getAccLen(pt_id)
                            L2 = self.snakelist[id - 1].length - L1
                            if pt_id!=0 and pt_id != self.snakelist[id-1].NP-1:
                                #if hit pt_id is in the middle of another artery, just branch the tail to collition
                                if L1 > minimum_length and L2 > minimum_length:
                                    if DEBUG:
                                        print('tail hit traced snake id',id-1,'add bif at pt', pt_id)
                                    self.snake[0].pos = self.snakelist[id-1][pt_id].pos
                                    self.snake[0].rad = self.snake[1].rad
                                    if self.collision == 2:
                                        self.collision = 3
                                    else:
                                        self.collision = 1
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                #else if hit near start of another artery, merge
                                elif L1 <= minimum_length and L2 > minimum_length and L3 > minimum_length:
                                    if DEBUG:
                                        print('tail hit traced snake id',id-1,'merge with snake', id-1, 'at L1 pt', pt_id)
                                    snake_backup = self.snake.copy()
                                    self.mergeSnake(self.snakelist[id-1].trimSnake(pt_id,True,True),reverse=False,append=False)
                                    if self.checkSharpTurn(angle_th):
                                        #no merging, just branching
                                        self.snake = snake_backup
                                        self.snake[0].pos = self.snakelist[id-1][pt_id].pos
                                        self.snake[0].rad = self.snake[1].rad
                                        if self.collision == 2:
                                            self.collision = 3
                                        else:
                                            self.collision = 1
                                        self.tail_collision_snake_id = id - 1
                                        return merging
                                    self.snakelist.valid_list[id-1] = False
                                    merging = True
                                    return merging
                                # else if hit near end of another artery, merge
                                elif L1 > minimum_length and L2 <= minimum_length and L3 > minimum_length:
                                    if DEBUG:
                                        print('tail hit traced snake id',id-1,'merge with snake', id-1, 'at L2 pt', pt_id)
                                    snake_backup = self.snake.copy()
                                    self.mergeSnake(self.snakelist[id-1].trimSnake(pt_id+1,False,True),reverse=True,append=False)
                                    if self.checkSharpTurn(angle_th):
                                        self.snake = snake_backup
                                        self.snake[0].pos = self.snakelist[id - 1][pt_id].pos
                                        self.snake[0].rad = self.snake[1].rad
                                        if self.collision == 2:
                                            self.collision = 3
                                        else:
                                            self.collision = 1
                                        self.tail_collision_snake_id = id - 1
                                        return merging
                                    self.snakelist.valid_list[id - 1] = False
                                    merging = True
                                    return merging
                                #L3 short, wait other branches merge with this small branch
                                #else:
                                    #if DEBUG:
                                    #    print('skip ptid',pt_id,'L123',L1,L2,L3)
                            elif pt_id==0 and L3>minimum_length:
                                if DEBUG:
                                    print('tail hit traced snake id',id-1,'merge with snake', id - 1, 'at 0 pt', pt_id)
                                snake_backup = self.snake.copy()
                                self.mergeSnake(self.snakelist[id - 1], reverse=False, append=False)
                                if self.checkSharpTurn(angle_th):
                                    # no merging, just branching
                                    self.snake = snake_backup
                                    self.snake[0].pos = self.snakelist[id - 1][pt_id].pos
                                    self.snake[0].rad = self.snake[1].rad
                                    if self.collision == 2:
                                        self.collision = 3
                                    else:
                                        self.collision = 1
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                self.snakelist.valid_list[id - 1] = False
                                merging = True
                                return merging
                            elif pt_id==self.snakelist[id-1].NP-1 and L3>minimum_length:
                                if DEBUG:
                                    print('tail hit traced snake id',id-1,'merge with snake', id - 1, 'at end pt', pt_id)
                                snake_backup = self.snake.copy()
                                self.mergeSnake(self.snakelist[id - 1], reverse=True, append=False)
                                if self.checkSharpTurn(angle_th):
                                    self.snake = snake_backup
                                    self.snake[0].pos = self.snakelist[id - 1][pt_id].pos
                                    self.snake[0].rad = self.snake[1].rad
                                    if self.collision == 2:
                                        self.collision = 3
                                    else:
                                        self.collision = 1
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                self.snakelist.valid_list[id - 1] = False
                                merging = True
                                return merging
                            #else:
                            #    if DEBUG:
                            #        print('skip ptid',pt_id,'L123',L1,L2,L3)
                        else:
                            self.snake[0].pos = self.snakelist[id - 1][pt_id].pos
                            self.snake[0].rad = self.snake[1].rad
                            merging = False
                            if self.collision == 2:
                                self.collision = 3
                            else:
                                self.collision = 1
                            self.tail_collision_snake_id = id - 1
                            return merging

        return merging

    def checkHeadCollision(self, index, collision_dist, minimum_length, automatic_merging,
                       max_angle, snake_id):
        DEBUG = 1
        angle_th = max_angle
        merging = False
        #collision 2: head, 3: both
        if self.collision == 2 or self.collision == 3:
            return merging
        SM = self.icafem.SM
        SN = self.icafem.SN
        SZ = self.icafem.SZ
        overlap = False

        tail_pt = index

        L3 = self.length
        for ix in range(-collision_dist,collision_dist+1):
            for iy in range(-collision_dist, collision_dist + 1):
                for iz in range(-collision_dist, collision_dist + 1):
                    temp_pt = Point3D(index.x+ix,index.y+iy,index.z+iz)
                    temp_pt.bound(SM,SN,SZ)
                    new_index = temp_pt.copy()
                    new_index.toIntPos()
                    id = int(self.icafem.getInt(new_index,'l'))
                    if id!=0 and id!=snake_id:
                        #check if snake is removed
                        if self.snakelist.valid_list[id-1]==False:
                            continue
                        overlap = True
                        #find nearest point at the traced snake
                        dist_temp = [tail_pt.dist(self.snakelist[id-1][i].pos) for i in range(self.snakelist[id-1].NP)]
                        pt_id = np.argmin(dist_temp)
                        if automatic_merging:
                            L1 = self.snakelist[id - 1].getAccLen(pt_id)
                            L2 = self.snakelist[id - 1].length - L1
                            if pt_id!=0 and pt_id != self.snakelist[id-1].NP-1:
                                #if hit pt_id is in the middle of another artery, just branch the tail to collition
                                if L1 > minimum_length and L2 > minimum_length:
                                    if DEBUG:
                                        print('head hit traced snake id',id-1,'add bif at pt', pt_id)
                                    self.snake[-1].pos = self.snakelist[id-1][pt_id].pos
                                    self.snake[-1].rad = self.snake[-2].rad
                                    if self.collision == 1:
                                        self.collision = 3
                                    else:
                                        self.collision = 2
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                #else if hit near start of another artery, merge
                                elif L1 <= minimum_length and L2 > minimum_length and L3 > minimum_length:
                                    if DEBUG:
                                        print('head hit traced snake id',id-1,'merge with snake', id-1, 'at L1 pt', pt_id)
                                    snake_backup = self.snake.copy()
                                    self.mergeSnake(self.snakelist[id-1].trimSnake(pt_id,True,True),reverse=False,append=True)
                                    if self.checkSharpTurn(angle_th):
                                        if DEBUG:
                                            print('sharp turn')
                                        #no merging, just branching
                                        self.snake = snake_backup
                                        self.snake[-1].pos = self.snakelist[id-1][pt_id].pos
                                        self.snake[-1].rad = self.snake[-2].rad
                                        if self.collision == 1:
                                            self.collision = 3
                                        else:
                                            self.collision = 2
                                        self.tail_collision_snake_id = id - 1
                                        return merging
                                    self.snakelist.valid_list[id-1] = False
                                    merging = True
                                    return merging
                                # else if hit near end of another artery, merge
                                elif L1 > minimum_length and L2 <= minimum_length and L3 > minimum_length:
                                    if DEBUG:
                                        print('head hit traced snake id',id-1,'merge with snake', id-1, 'at L2 pt', pt_id)
                                    snake_backup = self.snake.copy()
                                    self.mergeSnake(self.snakelist[id-1].trimSnake(pt_id+1,False,True),reverse=True,append=True)
                                    if self.checkSharpTurn(angle_th):
                                        self.snake = snake_backup
                                        self.snake[-1].pos = self.snakelist[id - 1][pt_id].pos
                                        self.snake[-1].rad = self.snake[-2].rad
                                        if self.collision == 1:
                                            self.collision = 3
                                        else:
                                            self.collision = 2
                                        self.tail_collision_snake_id = id - 1
                                        return merging
                                    self.snakelist.valid_list[id - 1] = False
                                    merging = True
                                    return merging
                                #else:
                                #    if DEBUG:
                                #        print('skip ptid',pt_id,'L123',L1,L2,L3)
                            elif pt_id==0 and L3 > minimum_length:
                                if DEBUG:
                                    print('head hit traced snake id',id-1,'merge with snake', id-1, 'at 0 pt', pt_id)
                                snake_backup = self.snake.copy()
                                self.mergeSnake(self.snakelist[id - 1], reverse=False, append=True)
                                if self.checkSharpTurn(angle_th):
                                    # no merging, just branching
                                    self.snake = snake_backup
                                    self.snake[-1].pos = self.snakelist[id - 1][pt_id].pos
                                    self.snake[-1].rad = self.snake[-2].rad
                                    if self.collision == 1:
                                        self.collision = 3
                                    else:
                                        self.collision = 2
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                self.snakelist.valid_list[id - 1] = False
                                merging = True
                                return merging
                            # else if hit near end of another artery, merge
                            elif pt_id == self.snakelist[id-1].NP-1 and L3 > minimum_length:
                                if DEBUG:
                                    print('head hit traced snake id',id-1,'merge with snake', id - 1, 'at end pt', pt_id)
                                snake_backup = self.snake.copy()
                                self.mergeSnake(self.snakelist[id - 1], reverse=True, append=True)
                                if self.checkSharpTurn(angle_th):
                                    self.snake = snake_backup
                                    self.snake[-1].pos = self.snakelist[id - 1][pt_id].pos
                                    self.snake[-1].rad = self.snake[-2].rad
                                    if self.collision == 1:
                                        self.collision = 3
                                    else:
                                        self.collision = 2
                                    self.tail_collision_snake_id = id - 1
                                    return merging
                                self.snakelist.valid_list[id - 1] = False
                                merging = True
                                return merging
                            #else:
                            #    if DEBUG:
                            #        print('skip ptid',pt_id,'L123',L1,L2,L3)
                        else:
                            self.snake[-1].pos = self.snakelist[id - 1][pt_id].pos
                            self.snake[-1].rad = self.snake[-2].rad
                            merging = False
                            if self.collision == 1:
                                self.collision = 3
                            else:
                                self.collision = 2
                            self.tail_collision_snake_id = id - 1
                            return merging
        return merging

    def _extractCNNTrackerPatch(self,pos):
        max_z = self.icafem.SZ
        max_y = self.icafem.SN
        max_x = self.icafem.SM

        cut_size = 9

        center_x_pixel = int(round(pos.x))
        center_y_pixel = int(round(pos.y))
        center_z_pixel = int(round(pos.z))

        left_x = center_x_pixel - cut_size
        right_x = center_x_pixel + cut_size
        left_y = center_y_pixel - cut_size
        right_y = center_y_pixel + cut_size
        left_z = center_z_pixel - cut_size
        right_z = center_z_pixel + cut_size

        new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))

        if not (left_x < 0 or right_x < 0 or left_y < 0 or right_y < 0 or \
                left_x >= max_x or right_x >= max_x or left_y >= max_y or \
                right_y >= max_y):
            for ind in range(left_z, right_z + 1):
                if ind < 0 or ind >= max_z:
                    continue
                src_temp = self.re_spacing_img[ind].copy()
                new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
            input_data = data_preprocess(new_patch)
            return input_data
        else:
            print('out of bound', left_x, right_x, left_y, right_y, left_z, right_z)
            return None

    def _processCNNTrackerResult(self, outputs_1, outputs_2, max_points, prev_dir, max_angle):
        outputs_1 = torch.nn.functional.softmax(outputs_1, 1)
        indexs = np.argsort(outputs_1.cpu().detach().numpy()[0])[::-1]
        curr_prob = prob_terminates(outputs_1, max_points).cpu().detach().numpy()[0]
        curr_r = outputs_2.cpu().detach().numpy()[0]
        sx, sy, sz = get_shell(max_points, curr_r)
        for i in range(len(indexs)):
            forward_move_direction = Point3D(sx[indexs[i]], sy[indexs[i]], sz[indexs[i]])
            if forward_move_direction.getAngleDeg(prev_dir) < max_angle:
                move_direction_x = sx[indexs[i]]
                move_direction_y = sy[indexs[i]]
                move_direction_z = sz[indexs[i]]
                return Point3D(move_direction_x, move_direction_y, move_direction_z), curr_r, curr_prob
        return Point3D(0, 0, 0), curr_r, curr_prob

    def CNNTrackerMoveNext(self,ht):
        #ht: head or tail
        assert self.NP > 2
        if ht=='h':
            pos = self.snake[-1].pos
        elif ht=='t':
            pos = self.snake[0].pos
        else:
            raise ValueError('undefined ht')

        max_points = 500
        max_angle = 99

        input_data = self._extractCNNTrackerPatch(pos)
        if input_data is None:
            return

        inputs = input_data.to(self.device)
        outputs = self.cnn_tracker_model(inputs.float())
        outputs = outputs.view((len(input_data), max_points + 1))
        outputs_1 = outputs[:, :len(outputs[0]) - 1]
        outputs_2 = outputs[:, -1]

        if ht=='h':
            prev_dir = self.snake[-1].pos - self.snake[-2].pos
        elif ht=='t':
            prev_dir = self.snake[0].pos - self.snake[1].pos

        next_direct, curr_r, curr_prob = self._processCNNTrackerResult(outputs_1, outputs_2, max_points, prev_dir, max_angle)
        return next_direct, curr_r, curr_prob