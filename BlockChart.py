import numpy as np
import cv2
from random import shuffle
from graphtools import *

global BlockSources
global sourceDict

class Sources():
    def __init__(self,insource = [], outsource=[]):
        self.ins = insource
        self.outs = outsource
        self.alldata = {}
        self.sourcelist = None
        updatefirst=True
    
    def setsources(self,insource, outsource):
        self.ins = insource
        self.outs = outsource
    
    def startsetup(self):
        for i in self.ins:
            i.findsource(self.sourcelist)
            i.pushdata()
            self.alldata.update(i.datasource["sdata"])
        for o in self.outs:
            o.findsource(self.sourcelist)
            o.pushdata()
            self.alldata.update(o.datasource["sdata"])
    
    def updateGet(self):
        for i in self.ins:
            i.loaddata()
            self.alldata.update(i.datasource["sdata"])

    def updateSet(self):
        for o in self.outs:
            o.datasource["sdata"]={k: self.alldata[k] for k in o.datasource["sdata"].keys()}
            o.pushdata()

            
class datasource():
    def __init__(self, sourcelabel="", source=None, sdata={}):
        self.datasource = {"sourcelabel":sourcelabel,\
                          "source":source,\
                          "sdata":sdata }
        
    def add(self,data):
        self.datasource["sdata"].update(data)
    
    def clear(self,data):
        self.datasource["sdata"].clear()
    
    def findsource(self, sourcelist):
        self.datasource["source"] = sourcelist[self.datasource["sourcelabel"]]
    
    def updatesource(self,label, sourcelist):
        self.datasource["sourcelabel"]=label
        self.findsource(sourcelist)

    def loaddata(self):
        self.datasource["sdata"]={k: self.datasource["source"].data[k] for k in self.datasource["sdata"].keys()}
    
    def pushdata(self):
        self.datasource["source"].data.update(self.datasource["sdata"])

#Basicfunctions:
#-BlockChart


def mouseaction(event, x,y,flags,param):
    MAIN_BC=param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        MAIN_BC.LBUTTONDOWN(x,y)

    if event == cv2.EVENT_LBUTTONUP:
        MAIN_BC.LBUTTONUP(x,y)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        MAIN_BC.LBUTTONDBLCLK(x,y)
    
    if event == cv2.EVENT_MOUSEMOVE:
        MAIN_BC.MOUSEMOVE(x,y)


class BlockChart():
    def __init__(self, FunctionBlocks,BlockSources,sourceDict, name = "BlockChart", image=np.ones((1600,1800,3), np.uint8)*255):
        self.image = image
        self.gpos = np.array([0,0])
        self.pos = np.array([0,0])
        self.connections =[]
        self.isRoot =True
        self.parent=None
        self.minimized = False
        #self.currentim = image.copy()
        self.FBs = FunctionBlocks
        self.marg=5
        self.highlightedFB=False
        self.hDX=np.array([0,0])
        self.bgcolor =(255,255,255)
        self.txtcolor = (0,0,0)
        #self.bus=Bus(label="MainBus")
        #self.FBs.append(FunctionBlock(self.bus,size=np.array([120,90]), pos=np.array([0,0])))

        for FB in self.FBs:
            FB.parent=self
            FB.Sources=Sources(BlockSources[FB.function.label]["In"],BlockSources[FB.function.label]["Out"])
            FB.Sources.sourcelist =sourceDict
            FB.Sources.startsetup()
            FB.Resize()
    
    def ResizeChart(self):
        for FB in self.FBs:
            FB.Resize()
        #self.arrangeChart(topcoll = False)
        BB=np.array([9999,0,9999,0])
        for FB in self.FBs:
            bb = FB.BoundingBox()
            if bb[0]<BB[0]: BB[0]=bb[0]
            if bb[1]>BB[1]: BB[1]=bb[1]
            if bb[2]<BB[2]: BB[2]=bb[2]
            if bb[3]>BB[3]: BB[3]=bb[3]
        BB=np.array([BB[0]-self.marg,BB[1]+self.marg,BB[2]-self.marg,BB[3]+self.marg,])
        for FB in self.FBs:
            bb = FB.BoundingBox()
            bb[:2]=bb[:2]-BB[0]
            bb[2:]=bb[2:]-BB[2]
            FB.SetBoundingBox(bb)
        BB[:2]=BB[:2]-BB[0]#+self.marg
        BB[2:]=BB[2:]-BB[2]+12
        self.image=np.ones((BB[3],BB[1],3), np.uint8)
        self.image[:,:,0]=self.bgcolor[0]
        self.image[:,:,1]=self.bgcolor[1]
        self.image[:,:,2]=self.bgcolor[2]
        
        return BB

    
    def deOverlapChart(self):
        arranged = False
        for i in range(len(self.FBs)-1):
            BB=self.FBs[i].BoundingBox()
            for FB in self.FBs[i+1:]:
                bb=FB.BoundingBox()
                ca= ((BB[0]<=bb[0]<=BB[1]) or (BB[0]<=bb[1]<=BB[1])) and \
                     ((BB[2]<=bb[2]<=BB[3]) or (BB[2]<=bb[3]<=BB[3]))
                cb= ((bb[0]<=BB[0]<=bb[1]) or (bb[0]<=BB[1]<=bb[1])) and \
                     ((bb[2]<=BB[2]<=bb[3]) or (bb[2]<=BB[3]<=bb[3]))
                if ca or cb:
                    sx=bb[1]-bb[0]
                    sy=bb[3]-bb[2]
                    #bb[0]=BB[1]+int(sx*0.1)+self.marg
                    #bb[1]=BB[1]+int(sx*1.1)+self.marg
                    bb[2]=BB[3]+int(sy*0.1)+self.marg
                    bb[3]=BB[3]+int(sy*1.1)+self.marg
                    FB.SetBoundingBox(bb)
                    arranged = True
        return arranged      

    def topCollapseChart(self):
        for FB in self.FBs:
            FB.posSet(np.array([FB.pos[1],0]))

    def nodeArrange(self):
        pass

    
    def arrangeChart(self, topcoll = True):
        if topcoll:
            self.topCollapseChart()
            arranged = True
        arranged = self.deOverlapChart()
        return arranged        
    
    def PotArrangeChart(self):#not used...
        for j in range(100):
            for i in self.connections:
                pot=2*np.array(potential(i[1].gpos-i[0].gpos),dtype = int)
                i[1].posSet(pot)
                i[0].posSet(-pot)
            k=0
            for FB in self.FBs:
                k+=1
                for FB2 in self.FBs[k:]:
                    pot=2*np.array(potential(FB.gpos-FB2.gpos),dtype = int)
                    FB.posSet(pot)
                    FB2.posSet(-pot)
                    
            
            #    i[0].posSet(np.array((i[0].gpos*2+i[1].gpos)/3-i[0].parent.gpos,dtype = int))
            #    i[1].posSet(np.array((i[1].gpos*2+i[0].gpos)/3-i[1].parent.gpos, dtype=int))
        self.ResizeChart()

    
    def updateChart(self):
        for FB in self.FBs:
            FB.update()       
    
    def DrawChart(self):
        self.connections = []
        self.updateChart()
        self.ResizeChart()
        im = self.image.copy()
        for FB in self.FBs:
            FB.drawFB(im)
        
        if self.isRoot:
            for i in self.connections:
                self.DrawConn(im,i[0],i[1],i[2],i[3], root = True)
            #self.PotArrangeChart()
        return im

    def DrawConn(self, im, obj_from, obj_to, color, keys, root =False):
        pos_from=obj_from.gpos - self.gpos
        pos_to=obj_to.gpos - self.gpos
        if ((all(pos_from == obj_from.pos) and all(pos_to == obj_to.pos)) or root):
            if pos_from[1]<pos_to[1]:
                pos_from[1]+=obj_from.size[1]
                pos_to[0] += obj_to.size[0]*0.55
                pos_from[0] += obj_from.size[0]*0.55
            else:
                pos_to[1]+=obj_to.size[1]
                pos_from[0] += obj_from.size[0]*0.45
                pos_to[0] += obj_to.size[0]*0.45

            cv2.circle(im,(int(pos_from[0]),int(pos_from[1])),3,color,-1)
            cv2.circle(im,(int(pos_to[0]),int(pos_to[1])),3,color,-1)
    
            cv2.line(im,(int(pos_from[0]),int(pos_from[1])),(int(pos_to[0]),int(pos_to[1])),color,1)
            cv2.putText(im,keys,(int(pos_from[0]*2/3+pos_to[0]/3),int(pos_to[1]/3+pos_from[1]*2/3) ),\
                        cv2.FONT_HERSHEY_SIMPLEX, .25, color)

            return True
        return False
        
    
    def FBinXY(self,x,y):
        for FB in self.FBs:
            bb=FB.BoundingBox()
            if (bb[0]<=x<=bb[1]) and (bb[2]<=y<=bb[3]):
                return FB
        return False

    def run(self,arr=True):    
        if arr:
            self.arrangeChart()
            self.ResizeChart()
            self.arrangeChart()
            self.ResizeChart()
        cv2.startWindowThread()
        cv2.namedWindow('BlockChart')
        cv2.setMouseCallback('BlockChart',mouseaction,self)
        while True:
            im=self.DrawChart()
            cv2.imshow("BlockChart",im)
            inp = cv2.waitKey(1)
            if inp==97:
                self.ResizeChart()
                self.arrangeChart()
                self.ResizeChart()
            if inp==27:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    def LBUTTONDOWN(self,x,y):
        FB = self.FBinXY(x,y)
        if not FB:
            return True
        if FB.LBUTTONDOWN(x,y):
            self.highlightedFB = FB
            self.hDX=self.highlightedFB.pos-np.array([x,y]) 
            return False
        return True

    def LBUTTONUP(self,x,y):
        self.highlightedFB = False
        FB = self.FBinXY(x,y)
        if not FB:
            return True
        return FB.LBUTTONUP(x,y)
       

    
    def LBUTTONDBLCLK(self,x,y):
        FB = self.FBinXY(x,y)
        if not FB:
            return False
        FB.LBUTTONDBLCLK(x,y)
        self.arrangeChart()
        self.ResizeChart()
        return True

    def MOUSEMOVE(self,x,y):
        if self.highlightedFB:
            self.highlightedFB.posSet(np.array([x,y])+self.hDX)
            return False
            #self.ResizeChart()
        FB = self.FBinXY(x,y)
        if not FB:
            return True
        return FB.MOUSEMOVE(x,y)



def potential(dX, rep=False):
    dx=dX[0]
    dy=dX[1]
    r2 = dx*dx + dy*dy
    if rep:
        p= -100.0/(1+r2)
    else:
        p= (r2/128.0-10.0)/(1.0+r2/400.0)/np.sqrt(r2+.0000001)
    return -p*np.array([dx+.1,dy+.1])
        

#-FunctionBlock
#-NullFunction, MainSwitch, Bus, SubChart
        
    
class FunctionBlock():
    def __init__(self, function, name = "NoName", size=np.array([120,90]), pos=np.array([120,90])):
        #basic info
        self.name =name
        self.size =size
        self.pos = pos
        self.gpos = pos
        #connections
        self.function=function
        self.function.parent=self 
        self.parent = None
        self.Sources = None
        #graphics
        self.image = None
        self.useimage = False
        self.textcolor = (0,0,0)
        self.bgcolor=(255,250,250)
        self.linecolor=(0,0,0)
        
    def update(self):
        self.Sources.updateGet()
        if hasattr(self.function, "update"):
            self.function.update()
        self.Sources.updateSet()
    
    def BoundingBox(self):
        return np.array([self.pos[0],self.pos[0]+self.size[0],self.pos[1],self.pos[1]+self.size[1]])
    
    def SetBoundingBox(self,bb):
        pos=np.array([bb[0],bb[2]])
        size=np.array([bb[1]-bb[0],bb[3]-bb[2]])
        self.posSet(pos)
        self.sizeSet(size)
    
    def posSet(self,pos):
        self.pos=pos
        self.gpos=self.parent.gpos + self.pos

    def sizeSet(self,size):
        self.size=size
    
    def Resize(self):
        if hasattr(self.function, "Resize"):
            self.function.Resize()

    def LBUTTONDOWN(self,x,y):
        if hasattr(self.function, "LBUTTONDOWN"):
            x=x-self.pos[0]
            y=y-self.pos[1]
            return self.function.LBUTTONDOWN(x,y)
        return True

    def LBUTTONUP(self,x,y):
        if hasattr(self.function, "LBUTTONUP"):
            x=x-self.pos[0]
            y=y-self.pos[1]
            return self.function.LBUTTONUP(x,y)
        return True

    def MOUSEMOVE(self,x,y):
        if hasattr(self.function, "MOUSEMOVE"):
            x=x-self.pos[0]
            y=y-self.pos[1]
            return self.function.MOUSEMOVE(x,y)
        return True

    def LBUTTONDBLCLK(self,x,y):
        if hasattr(self.function, "LBUTTONDBLCLK"):
            x=x-self.pos[0]
            y=y-self.pos[1]
            return self.function.LBUTTONDBLCLK(x,y)
        return True

    def drawFB(self, im):
        self.posSet(self.pos)
        cv2.rectangle(im, (self.pos[0], self.pos[1]), \
                      (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), self.bgcolor, -1)
        cv2.rectangle(im, (self.pos[0], self.pos[1]), \
                      (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), self.linecolor, 2)
        if hasattr(self.function,"presentData"):
            cv2.putText(im,self.function.presentData(),\
                        (self.pos[0]+int(self.size[0]*.1), self.pos[1] + 30),\
                        cv2.FONT_HERSHEY_SIMPLEX, .25, self.textcolor)
        if hasattr(self.function, "Draw"):
            self.function.Draw(im)
        for i in self.Sources.ins:
            color = (200,100,100)

            self.pushToConnections(i,color)
        for i in self.Sources.outs:
            color = (50,100,50)
            self.pushToConnections(i,color,indirection = False)

    def pushToConnections(self,i,color,indirection = True):
            if not (i.datasource["source"].parent == self):
                ds=i.datasource["source"].parent
                keys=str(list(i.datasource["sdata"].keys()))[1:-1]
                if ds.parent.minimized:
                    ds=ds.parent.parent.parent
                if indirection:
                    self.parent.connections.append([self,ds,color,keys])
                else:
                    self.parent.connections.append([ds,self,color,keys])
                if self.function.type == "SubChart":
                    if i.datasource["source"].parent.parent == self.function.chart:
                        self.parent.connections.pop()
        

    
class OnOff():
    def __init__(self, onstate=False, label ="On/Off"):
        self.type = "On/Off"
        self.label=label
        self.onstate=onstate
        self.parent = None
    
    def update(self):
        self.parent.Sources.alldata["Onstate"] =self.onstate
    
    def swithc(self):
        self.onstate=not self.onstate
    
    def Resize(self):
        self.parent.sizeSet(np.array([120,35]))    

    def Draw(self,im):
        pass

    def LBUTTONDOWN(self,x,y):
        self.onstate = not self.onstate
        return True

    def LBUTTONUP(self,x,y):
        return True
        
        
    def MOUSEMOVE(self,x,y):
        return True
        

    def LBUTTONDBLCLK(self,x,y):
        return True

    def presentData(self):
        if self.onstate:
            state="On"
        else:
            state="Off"
        return self.label+": "+state #tekstiä

class Bus():
    def __init__(self, busdata={"Round":0}, label = "DataBus"):
        self.type = "Bus"
        self.label = label
        self.data = busdata #dictionary
        self.parent = None    
    
    def Resize(self):
        self.parent.size = np.array([400,35])    
 
    def presentData(self):
        return self.label+": "+str(self.data) #tekstiä

    
class SubChart():
    def __init__(self, subchart, label="SubChart"):
        self.type = "SubChart"
        self.label =label
        self.chart=subchart
        self.chart.isRoot =False
        self.chart.parent=self
        #self.minimized = False
        self.parent = None
    
    def update(self):
        self.chart.updateChart()

    def Draw(self,im):
        #self.chart.bgcolor = self.parent.bgcolor
        pos=self.parent.pos
        size=self.parent.size
        self.chart.gpos= self.parent.gpos
        #im[pos[1]:pos[1]+size[1],pos[0]:pos[0]+size[0],:] = self.chart.DrawChart()
        if not self.chart.minimized:
            im[pos[1]:pos[1]+size[1],pos[0]:pos[0]+size[0],:] = self.chart.DrawChart()
            cv2.putText(im,self.presentData(),(pos[0]+2, pos[1] +size[1]-2),cv2.FONT_HERSHEY_SIMPLEX, .5, self.parent.textcolor)
        for i in self.chart.connections:
            if self.chart.minimized:
                if i[0].parent == self.chart:
                    i[0]=self.parent
                if i[1].parent == self.chart:
                    i[1]=self.parent
                if not(i[0]==i[1]):# and not(i[0]==i[1].parent) and not(i[1]==i[0].parent)):
                    self.parent.parent.connections.append(i)
            else:
                self.parent.parent.connections.append(i)

    def Resize(self):
        if not self.chart.minimized:
            #self.chart.arrangeChart()
            BB = self.chart.ResizeChart()
            self.parent.sizeSet(np.array([BB[1],BB[3]]))
        else:
            self.parent.sizeSet(np.array([120,35]))

    def LBUTTONDOWN(self,x,y):
        if not self.chart.minimized:
            return self.chart.LBUTTONDOWN(x,y)
        return True

    def LBUTTONUP(self,x,y):
        return self.chart.LBUTTONUP(x,y)

    def MOUSEMOVE(self,x,y):
        return self.chart.MOUSEMOVE(x,y)


    def LBUTTONDBLCLK(self,x,y):
        if self.chart.minimized:
            self.chart.minimized = not self.chart.minimized
            self.Resize()
            return True
        if not self.chart.LBUTTONDBLCLK(x,y):
            self.chart.minimized = not self.chart.minimized
            self.Resize()
            return True
            
    def presentData(self):
        return self.label #tekstiä


class Monitor():  #Template function
    def __init__(self, label="Monitor"):
        self.type = "Monitor"
        self.label =label
        self.parent = None
        self.deltatime=60.0 #sekunteina 
        self.MonitorData =[]
        self.monitorImage = None
        self.h=150
        self.w=400
        self.sourcelabels =[]#["V_PV","E_SC","Lightness"]
        self.minimized =False
    
    def update(self):
        dt=self.deltatime #vaihdetaan logiikan pyytämään aikahyppyyn myöhemmin
        self.sourcelabels = []
        data=[]
        for i in self.parent.Sources.ins:
            lab = list(i.datasource['sdata'].keys())
            for l in lab:
                self.sourcelabels.append(l)
                data.append(self.parent.Sources.alldata[l])
        #V_PV=self.parent.Sources.alldata["V_PV_Out"]
        #E_SC=self.parent.Sources.alldata["E_SC"]
        #lightness = self.parent.Sources.alldata["Lightness"]
        self.MonitorData.append(data)#[V_PV,E_SC,lightness]) 

    def Draw(self,im):
        if not self.minimized:
            data=np.array(self.MonitorData[-self.w:])
            ld=len(data[0])
            mxd=list(np.zeros(ld))
            for i in range(ld):
                mxd[i]=max(data[:,i])
                #mni=min(data[:,i])
                if mxd[i]>0:
                    data[:,i]=self.h-data[:,i]/mxd[i]*self.h
            #if len(data)>300: data[:,:300]=data[:,-300:]
            img=np.zeros((self.h,self.w,3), np.uint8)
        
            for j in range(ld):
                #for i
                #cv2.polylines(img, np.int32([points]), 1, (255,255,255))
                for i in range(len(data)-1): 
                    color= (int(127+128*np.sin(6.0*j/ld)),int(127+128*np.cos(6.0*j/ld)),int(127-128*np.cos(6.0*j/ld)))      
                    cv2.line(img,(i,int(data[i,j])),(i+1,int(data[i+1,j])),color,1)
                    cv2.putText(img,self.sourcelabels[j]+" MX: "+str(np.round(mxd[j],2)),\
                        (5,10+j*15),\
                        cv2.FONT_HERSHEY_SIMPLEX, .25, color )

            im[self.parent.pos[1]:self.parent.pos[1]+self.h,self.parent.pos[0]:self.parent.pos[0]+self.w]=img
        

    def Resize(self):
        self.parent.textcolor = (255,200,200)
        if self.minimized:
            self.parent.sizeSet(np.array([120,90]))
            self.parent.bgcolor=(50,0,0)
        else:
            self.parent.sizeSet(np.array([self.w,self.h]))    

    #def LBUTTONDOWN(self,x,y):
    #    return True
        
    #def LBUTTONUP(self,x,y):
    #    return True
        
    def LBUTTONDBLCLK(self,x,y):
        self.minimized=not self.minimized
        return True
    
    #def MOUSEMOVE(self,x,y):
    #    return True
    
    def presentData(self):
        return self.label #tekstiä

class NullFunc():  #Template function
    def __init__(self, label="NullFunc"):
        self.type = "NullFunc"
        self.label =label
        self.parent = None
    
    #def update(self):
    #    pass

    #def Draw(self,im):
    #    pass

    def Resize(self):
        self.parent.sizeSet(np.array([120,35]))    

    #def LBUTTONDOWN(self,x,y):
    #    return True
        
    #def LBUTTONUP(self,x,y):
    #    return True
        
    #def LBUTTONDBLCLK(self,x,y):
    #    return True
    
    #def MOUSEMOVE(self,x,y):
    #    return True
    
    def presentData(self):
        return self.label #tekstiä
