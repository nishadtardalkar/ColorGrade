import cv2
from numba import cuda
import numpy as np

class ColorGrade():

    def __init__(self, tonemap_preset='ACES'):

        # TONEMAP SETTINGS
        self.tonemap_preset = tonemap_preset
        if tonemap_preset == 'ACES':
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
        elif tonemap_preset == 'NONE':
            a = b = c = d = e = 0
        self.tonemap_coeffs = np.array([a,b,c,d,e], dtype=np.float32)
        
        # COLOR CORRECTION SETTINGS
        self.cc_r_res = 32
        self.cc_g_res = 32
        self.cc_b_res = 32
        self.cc_vals = np.zeros(self.cc_r_res + self.cc_g_res + self.cc_b_res, dtype=np.float32)  
        self.cc_res = np.array([self.cc_r_res, self.cc_g_res, self.cc_b_res])     
        
    @cuda.jit
    def k_tonemap(img, coeffs, size):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= size:
            return

        a = coeffs[0]
        b = coeffs[1]
        c = coeffs[2]
        d = coeffs[3]
        e = coeffs[4]
        
        v = img[i]
        v = v*(a*v+b)/(v*(c*v+d)+e)
        img[i] = v

         
    @cuda.jit
    def k_colorcorrect(img, res, vals, size):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= size:
            return

        v = img[i]
        s = 0
        m = 2-i%3
        if m == 0:
            c = 0
        elif m == 1:
            c = res[0]
        elif m == 2:
            c = res[0]+res[1]
        for j in range(res[m]):
            p = j/res[m]
            s += (1-abs(v-p))*vals[c+j]
        s = max(min(v+s, 1), 0)
        img[i] = s

        
    @cuda.jit
    def k_channelcontour(points, vals, res, size):
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= size:
            return

        v = points[i]
        s = 0
        for j in range(res):
            p = j/res
            s += (1-abs(v-p))*vals[j]
        s = max(min(v+s, 1), 0)
        points[i] = s

        
    def set_tonemap_coeffs(self, a, b, c, d, e):
        self.tonemap_coeffs = np.array([a,b,c,d,e], dtype=np.float32)

    def set_cc_resolution(self, channel, res):
        if channel == 'R':
            vals = np.zeros(res + self.cc_g_res + self.cc_b_res, dtype=np.float32)
            for i in range(res):
                p = i*self.cc_r_res/res
                vals[i] = self.cc_vals[int(p)]*(1-(p-int(p))) + self.cc_vals[int(p)+1]*(p-int(p))
            vals[res:] = self.cc_vals[self.cc_r_res:]
            self.cc_r_res = res
            self.cc_vals = vals
        elif channel == 'G':
            vals = np.zeros(self.r_res + res + self.cc_b_res, dtype=np.float32)
            for i in range(res):
                p = i*self.cc_g_res/res
                vals[self.cc_r_res+i] = self.cc_vals[int(p)+self.cc_r_res]*(1-(p-int(p))) + self.cc_vals[int(p)+1+self.cc_r_res]*(p-int(p))
            vals[:self.r_res] = self.cc_vals[:self.r_res]
            vals[self.r_res+res:] = self.cc_vals[self.r_res+self.g_res:]
            self.cc_g_res = res
            self.cc_vals = vals
        elif channel == 'B':
            vals = np.zeros(self.r_res + res + self.cc_b_res, dtype=np.float32)
            for i in range(res):
                p = i*self.cc_b_res/res
                vals[self.cc_r_res+self.cc_g_res+i] = self.cc_vals[int(p)+self.cc_r_res+self.cc_g_res]*(1-(p-int(p))) + self.cc_vals[int(p)+1+self.cc_r_res+self.cc_g_res]*(p-int(p))
            vals[:-res] = self.cc_vals[:-self.b_res]
            self.cc_b_res = res
            self.cc_vals = vals
        self.cc_res = np.array([self.cc_r_res, self.cc_g_res, self.cc_b_res])

    def add_cc_val(self, channel, p, val):
        if channel == 'R':
            i = min(max(int(p*self.cc_r_res), 0), self.cc_r_res)
            self.cc_vals[i] += val
        elif channel == 'G':
            i = min(max(int(p*self.cc_g_res), 0), self.cc_g_res)
            self.cc_vals[self.cc_r_res+i] += val
        elif channel == 'B':
            i = min(max(int(p*self.cc_b_res), 0), self.cc_b_res)
            self.cc_vals[self.cc_r_res+self.cc_g_res+i] += val

    def set_size(self, h, w):
        self.h = h
        self.w = w
        self.framesize = h*w*3
        self.blocksPGrid = int(np.ceil(self.framesize/1024.0))
        if self.blocksPGrid > 1:
            self.threadsPBlock = 1024
        else:
            self.threadsPBlock = self.framesize

    def get_cc_contour(self, frame, channel):
        alpha = 0.001
        overlay = frame.copy()
        xlen = int(frame.shape[1]*0.8)
        self.blocks = int(np.ceil(xlen/1024.0))
        self.threads = 1024 if self.blocks > 1 else xlen
        points = np.arange(xlen)/xlen
        d_points = cuda.to_device(points)
        if channel == 'R':
            for i in range(self.cc_r_res):
                x = int(i*frame.shape[1]*0.8/self.cc_r_res + frame.shape[1]*0.1)
                y = int(-(self.cc_vals[i]*frame.shape[0]*0.25) + frame.shape[0]*0.5)
                overlay = cv2.rectangle(overlay, (x-2, y-2), (x+2, y+2), (0,0,255), -1)
            ColorGrade.k_channelcontour[(self.blocks),(self.threads)](d_points, self.cc_vals[:self.cc_r_res], self.cc_r_res, xlen)
            points = frame.shape[0]-d_points.copy_to_host()*frame.shape[0]*0.6-frame.shape[0]*0.2
            polyline = np.zeros((xlen, 2), dtype=np.int32)
            polyline[:,0] = np.arange(xlen)+int(frame.shape[1]*0.1)
            polyline[:,1] = points
            overlay = cv2.polylines(overlay, [polyline], False, (0,0,255), 2)
            
        elif channel == 'G':
            for i in range(self.cc_g_res):
                x = int(i*frame.shape[1]*0.8/self.cc_g_res + frame.shape[1]*0.1)
                y = int(-(self.cc_vals[self.cc_r_res+i]*frame.shape[0]*0.25) + frame.shape[0]*0.5)
                overlay = cv2.rectangle(overlay, (x-2, y-2), (x+2, y+2), (0,255,0), -1)
            ColorGrade.k_channelcontour[(self.blocks),(self.threads)](d_points, self.cc_vals[self.cc_r_res:-self.cc_b_res], self.cc_g_res, xlen)
            points = frame.shape[0]-d_points.copy_to_host()*frame.shape[0]*0.6-frame.shape[0]*0.2
            polyline = np.zeros((xlen, 2), dtype=np.int32)
            polyline[:,0] = np.arange(xlen)+int(frame.shape[1]*0.1)
            polyline[:,1] = points
            overlay = cv2.polylines(overlay, [polyline], False, (0,255,0), 2)
            
        elif channel == 'B':
            for i in range(self.cc_b_res):
                x = int(i*frame.shape[1]*0.8/self.cc_b_res + frame.shape[1]*0.1)
                y = int(-(self.cc_vals[self.cc_r_res+self.cc_g_res+i]*frame.shape[0]*0.25) + frame.shape[0]*0.5)
                overlay = cv2.rectangle(overlay, (x-2, y-2), (x+2, y+2), (255,0,0), -1)
            ColorGrade.k_channelcontour[(self.blocks),(self.threads)](d_points, self.cc_vals[-self.cc_b_res:], self.cc_b_res, xlen)
            points = frame.shape[0]-d_points.copy_to_host()*frame.shape[0]*0.6-frame.shape[0]*0.2
            polyline = np.zeros((xlen, 2), dtype=np.int32)
            polyline[:,0] = np.arange(xlen)+int(frame.shape[1]*0.1)
            polyline[:,1] = points
            overlay = cv2.polylines(overlay, [polyline], False, (255,0,0), 2)
            
        frame = overlay*alpha + frame*(1-alpha)
        return frame
        

    def apply(self, frame):
        d_frame = cuda.to_device(frame.flatten().astype(np.float32))
        ColorGrade.k_tonemap[(self.blocksPGrid),(self.threadsPBlock)](d_frame, self.tonemap_coeffs, np.int32(self.framesize))
        ColorGrade.k_colorcorrect[(self.blocksPGrid),(self.threadsPBlock)](d_frame, self.cc_res, self.cc_vals, np.int32(self.framesize))
        return d_frame.copy_to_host().reshape(self.h, self.w, 3)

