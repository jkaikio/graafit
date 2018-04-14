import cv2
import numpy as np
from random import shuffle



#apufunktiot########################################################
def potential(dx,dy, mode="normal", scale=1.0):
    #dx=dX[0]
    #dy=dX[1]
    r2 = dx*dx + dy*dy
    r=np.sqrt(r2)
    dX=Normalize(np.array([dx,dy]))
    if mode=="rep":
        p= -20.0/(1+r2)
    if mode=="normal":
        p= (r2/128.0-10.0)/(1.0+r2/400.0)/(r+.001)

    if mode=="grouprep":
        p= -100.0/(1+r2)
    if mode=="grouppull":
        p= (r2/128.0-10.0)/(1.0+r2/400.0)/(r+.001)
        p*=10
        if p<0: p=0
    return -p*np.array([dx,dy])*scale


def RectPotential(n1,n2, mode="normal", scale=1):
    collx= False
    if n2.x>n1.x:
        dx=(n2.x-n2.sx)-(n1.x+n1.sx)
        if dx < 0: collx=True
    else:
        dx=(n2.x+n2.sx)-(n1.x-n1.sx)
        if dx > 0: collx = True

    colly= False
    if n2.y>n1.y:
        dy=(n2.y-n2.sy)-(n1.y+n1.sy)
        if dy < 0: colly=True
    else:
        dy=(n2.y+n2.sy)-(n1.y-n1.sy)
        if dy > 0: colly = True
    
    if mode=="inside":
        colly = not colly
        collx = not collx
        #dx= -dx
        #dy= -dy
        #mode="normal"
    dX=Normalize(np.array([dx,dy]))
    
    if collx and colly:
        if np.abs(dx)>np.abs(dy):
            return np.array([-dx/2,0])
        else:
            return np.array([0,-dy/2])
    #    p=5*np.array([dx,dy])
    #    return p
    #if collx:
    #    dx=0    
    #if colly:
    #    dy=0
    
        
    r2 = dx*dx + dy*dy
    p=(r2*.1)/(10+r2) 
    return -p*dX*scale
    '''
    if mode=="rep":
        p= -20.0/(1+r2)
    if mode=="normal":
        p= (r2/128.0-10.0)/(1.0+r2/400.0)/np.sqrt(r2+.0000001)

    if mode=="grouprep":
        p= -100.0/(1+r2)
    if mode=="grouppull":
        p= (r2/128.0-10.0)/(1.0+r2/400.0)/np.sqrt(r2+.0000001)
        p*=10
        if p<0: p=0
    '''
def Normalize(vect):
    x=vect[0]
    y=vect[1]
    if (y==0) and (x==0):
        v=np.array([0,0])
    else:
        r=np.sqrt(x*x+y*y)
        v=np.array([x/r, y/r])
    return v
#########################################################################################

class node():
    def __init__(self,x,y,sx,sy):
        self.x=x
        self.y=y
        self.sx=sx/2
        self.sy=sy/2
        self.pot = np.array([0,0])
        self.conn =[]
        self.color =(0,0,0)
        self.cargo = None
        self.type = "Node"
        
    def connSet(self, conn):
        self.conn = conn
    
    def drawNode(self,im, r=3,colorconn=(150,150,150), scale=1, \
                 cent=np.array([0,0]), connections=True):
        x= int((self.x-cent[0])*scale)
        y= int((self.y-cent[1])*scale)
        sx= int(self.sx*scale)
        sy= int(self.sy*scale)
        
        cv2.circle(im, (x,y),r,self.color,1)
        cv2.rectangle(im,(x-sx,y-sy),(x+sx,y+sy),self.color,1)
        
        
        
        if connections:
            for c in self.conn:
                colorconn=(int(self.color[0]/2+c.color[0]/2),\
                           int(self.color[1]/2+c.color[1]/2),\
                           int(self.color[2]/2+c.color[2]/2))
                #print(colorconn)
                xc= int((c.x-cent[0])*scale)
                yc= int((c.y-cent[1])*scale)
                sxc= int(c.sx*scale)
                syc= int(c.sy*scale)
    
                cv2.line(im,(x,y),(xc,yc),colorconn,1)
#######################################################################################################        
class nodegroup():
    def __init__(self,x,y,sx,sy,nodes):
        self.x=x
        self.y=y
        self.sx=sx/2
        self.sy=sy/2
        self.nodes =nodes
        self.conn = []
        self.updateConn()
        self.pot = np.array([0,0])
        self.color =(200,200,200)
        self.cargo=None
        self.parent=None
        self.type = "Nodegroup"

    def updateConn(self):
        self.conn=[]
        for n in self.nodes:
            for c in n.conn:
                self.conn.append(c)
    
    def BoundingBoxSet(self,resize=False):
        BB=np.array([999999999,-999999999,999999999,-999999999])
        for n in self.nodes:
            if BB[0] > (n.x-n.sx): BB[0] = n.x-n.sx
            if BB[1] < (n.x+n.sx): BB[1] = n.x+n.sx
            if BB[2] > (n.y-n.sy): BB[2] = n.y-n.sy
            if BB[3] < (n.y+n.sy): BB[3] = n.y+n.sy
        if resize:
            self.x=BB[0]/2+BB[1]/2
            self.y=BB[2]/2+BB[3]/2
            self.sx=(BB[1]-BB[0])/2
            self.sy=(BB[3]-BB[2])/2
        return BB
        
    def collapse(self,direction,backwards=False):
        self.BoundingBoxSet(resize=True)
        
        d=Normalize(direction)
        drx=1
        dry=1
        if d[0]<0:
            drx=-1
        if d[1]<0:
            dry=-1

        dist=[]
        for n in self.nodes:
            #X=np.array([n.x+drx*n.sx-self.x-drx*self.sx,n.y+dry*n.sy-self.y-dry*self.sy])
            X=np.array([n.x-self.x,n.y-self.y])
            if backwards:
                dist.append([np.dot(X,d),n])
            else:
                dist.append([np.dot(X,-1*d),n])
        dist=sorted(dist)


        gx=self.x+drx*self.sx
        gy=self.y+dry*self.sy
        
        i=0
        for n in dist:
            nx = n[1].x+drx*n[1].sx
            ny = n[1].y+dry*n[1].sy
            
            dx=gx-nx
            dy=gy-ny
            
            if d[0]==0: tx=9999999999
            else: tx = dx/d[0]
            if d[1]==0: ty=9999999999
            else: ty = dy/d[1]
            
            if tx<=ty: 
                t=tx
                xdir=True
            else: 
                t=ty
                xdir=False
            T=t
            if backwards:
                i_e=-1
                i_s=i
            else:
                i_e=i
                i_s=0
            for nn in dist[i_s:i_e]:        
                nnx = nn[1].x-drx*nn[1].sx
                nny = nn[1].y-dry*nn[1].sy
    
                dx=nnx-nx
                dy=nny-ny
            
                if d[0]==0: tx=0
                else: tx = dx/d[0]
                if d[1]==0: ty=0
                else: ty = dy/d[1]
            
                if tx>=ty: 
                    t=tx
                    xndir=True
                else: 
                    t=ty
                    xndir=False
                
                if t<T:
                    T=t
                    xdir=xndir
                
            n[1].x=n[1].x+d[0]*T
            n[1].y=n[1].y+d[1]*T

            #toinen suunta
            xdir = not xdir
            nx = n[1].x+drx*n[1].sx
            ny = n[1].y+dry*n[1].sy
            
            dx=gx-nx
            dy=gy-ny

            if xdir:
                t=drx*dx
            else: 
                t=dry*dy
            T=t
                
            for nn in dist[i_s:i_e]:        
                nnx = nn[1].x-drx*nn[1].sx
                nny = nn[1].y-dry*nn[1].sy
    
                dx=nnx-nx
                dy=nny-ny
                
                t=999999999
                if xdir:
                    if  (nn[1].y-nn[1].sy <=  n[1].y-n[1].sy  <= nn[1].y+nn[1].sy) or\
                        (nn[1].y-nn[1].sy <=  n[1].y+n[1].sy  <= nn[1].y+nn[1].sy) or\
                        (n[1].y-n[1].sy   <= nn[1].y-nn[1].sy <=   n[1].y+n[1].sy) or\
                        (n[1].y-n[1].sy   <= nn[1].y+nn[1].sy <=   n[1].y+n[1].sy):
                        t=drx*dx    
                else: 
                    if  (nn[1].x-nn[1].sx <=   n[1].x-n[1].sx <= nn[1].x+nn[1].sx) or\
                        (nn[1].x-nn[1].sx <=   n[1].x+n[1].sx <= nn[1].x+nn[1].sx) or\
                        ( n[1].x-n[1].sx   <= nn[1].x-nn[1].sx <= n[1].x+n[1].sx ) or\
                        ( n[1].x-n[1].sx   <= nn[1].x+nn[1].sx <= n[1].x+n[1].sx ):
                        t=dry*dy
                if 0<t<T:T=t
                
            if xdir:    
                n[1].x=n[1].x+drx*T
            else:
                n[1].y=n[1].y+dry*T            
            
            i+=1
        self.BoundingBoxSet(resize=True)
    
    def setColor(self,color, nodes=True):
        self.color=color
        if nodes:
            for n in self.nodes:
                n.color=color
    
    def drawNode(self,im, r=3,colorconn=(150,150,150), scale=1.0, \
                 cent=np.array([0,0]), connections=True):
        self.drawNodeGroup(im, r, scale, cent)

    def drawNodeGroup(self,im, r=3, scale=1.0, cent=np.array([0,0])):
        x= int((self.x-cent[0])*scale)
        y= int((self.y-cent[1])*scale)
        sx= int(self.sx*scale)
        sy= int(self.sy*scale)
        
        cv2.circle(im, (x,y),r,self.color,-1)
        cv2.rectangle(im,(x-sx,y-sy),(x+sx,y+sy),self.color,1)

#################################################################################################
class graph():
    def __init__(self,nodes, nodegroups=[]):
        self.nodes = nodes
        self.nodegroups = nodegroups
        self.BB = None
        self.BoundingBoxSet() 
        self.imgsize = (500,500,3)
    
    def BoundingBoxSet(self):
        BB=np.array([999999999,-999999999,999999999,-999999999])
        for n in self.nodes:
            if BB[0] > (n.x-n.sx): BB[0] = n.x-n.sx
            if BB[1] < (n.x+n.sx): BB[1] = n.x+n.sx
            if BB[2] > (n.y-n.sy): BB[2] = n.y-n.sy
            if BB[3] < (n.y+n.sy): BB[3] = n.y+n.sy
        for n in self.nodegroups:
            if BB[0] > (n.x-n.sx): BB[0] = n.x-n.sx
            if BB[1] < (n.x+n.sx): BB[1] = n.x+n.sx
            if BB[2] > (n.y-n.sy): BB[2] = n.y-n.sy
            if BB[3] < (n.y+n.sy): BB[3] = n.y+n.sy
        self.BB=BB
        return self.BB
            
        
    def DrawGraph(self):
        img=np.ones(self.imgsize, dtype = np.uint8)*255
        
        BB=self.BoundingBoxSet()
        c=np.array([0,0])
        c[0]=BB[0]
        c[1]=BB[2]
        scx=self.imgsize[0]/(BB[1]-BB[0])
        scy=self.imgsize[1]/(BB[3]-BB[2])
        sc=min(scx,scy)
        
        for n in self.nodes:
            n.drawNode(img,scale=sc, cent=c)
        for N in self.nodegroups:
            N.drawNodeGroup(img,scale=sc,cent=c)
        return img
    
    def PotStep(self, n, nn, mode, scale=1, potstyle="point",onesided=False):
        if potstyle =="point":
            pot=potential(nn.x-n.x, nn.y-n.y, mode=mode, scale=scale)
        if potstyle =="rect":
            pot=RectPotential(n, nn, mode=mode, scale=scale)
        nn.x+=pot[0]
        nn.y+=pot[1]
        if not onesided:
            n.x-=pot[0]
            n.y-=pot[1]
        #return -pot,pot
                
    def StepPotNodePoints(self,sc=1):
        i=0
        for n in self.nodes[:-2]:
            i+=1
            if n.type == "Node":
                for nn in self.nodes[i:]:
                    if nn.type == "Node":
                        self.PotStep(n, nn, "rep",scale=sc)
        for n in self.nodes:
            if n.type == "Node":
                for nn in n.conn:
                    if nn.type == "Node":
                        self.PotStep(n, nn, "normal",scale=sc)                
        
    def StepPotNodegroupsPoints(self):
        for g in self.nodegroups:
            for n in g.nodes:
                self.PotStep(g, n, "grouppull")
        #i=0
        #for g in self.nodegroups[:-2]:
        #    i+=1
        #    for gg in self.nodegroups[i:]:
        #        if (not(gg.parent==g)) and (not (g.parent==gg)):
        #            self.PotStep(g, gg, "grouprep")
    
    def StepPotNodeRect(self,sc=1):## ei päivitetty if-node-parent-mielessä
        i=0
        for n in self.nodes[:-2]:
            i+=1
            for nn in self.nodes[i:]:
                self.PotStep(n, nn, "normal",potstyle="rect")
        #for n in self.nodes:
        #    for nn in n.conn:
        #        self.PotStep(n, nn, "normal",potstyle="rect")                
        
    def StepPotNodegroupsRect(self):## ei päivitetty if-node-parent-mielessä
        for g in self.nodegroups:
            for n in g.nodes:
                self.PotStep(g, n, "inside", potstyle="rect", onesided=True)
        i=0
        for g in self.nodegroups[:-2]:
            i+=1
            for gg in self.nodegroups[i:]:
                self.PotStep(g, gg, "normal", potstyle="rect")
     
    def PotNodePoints(self,sc=1,sc2=1,r_pot=False):
        i=0
        for n in self.nodes[:-2]:
            if n.type=="Node":
                if r_pot: n.pot=np.array([0,0])
                i+=1
                for nn in self.nodes[i:]:
                    if n.type=="Node":
                        pot=potential(nn.x-n.x, nn.y-n.y, mode="rep", scale=sc)
                        n.pot-=pot
                        nn.pot+=pot
        for n in self.nodes:
            for nn in n.conn:
                pot=potential(nn.x-n.x, nn.y-n.y, mode="normal",scale=sc2)
                n.pot-=pot
                nn.pot+=pot
                
    def PotNodegroupsPoints(self, r_pot = False):
        for g in self.nodegroups:
            if r_pot: g.pot=np.array([0,0])
            for n in g.nodes:
                pot=potential(n.x-g.x, n.y-g.y, mode="grouppull")
                g.pot-=pot
                n.pot+=pot
        #i=0
        #for g in self.nodegroups[:-2]:
        #    i+=1
        #    for gg in self.nodegroups[i:]:
        #        if (not(gg.parent==g)) and (not (g.parent==gg)):
        #            pot=potential(gg.x-g.x, gg.y-g.y, mode="rep")
        #            g.pot-=pot
        #            n.pot+=pot
                
    def ClearPot(self):
        for g in self.nodegroups:
            g.pot=np.array([0.0,0.0])
        for n in self.nodes:
            n.pot=np.array([0.0,0.0])
    
    def StepAllPot(self, ngs=True, ns=True, gps_in=False):
        if ngs:
            i=0
            for g in self.nodegroups[:-2]:
                #i+=1
                #for gg in self.nodegroups[i:]:
                #    if (not(gg.parent==g)) and (not (g.parent==gg)): 
                #        self.Collide(g,gg)
                if gps_in:
                    for n in g.nodes:
                        self.Collide(g,n,inside=True)
                    j=0
                    for n in g.nodes[:-2]:
                        j+=1
                        for nn in self.nodes[j:]:
                            self.Collide(n,nn)
            for g in self.nodegroups:
                g.x+=g.pot[0]
                g.y+=g.pot[1]
      
        if ns:
        #    i=0
        #    for n in self.nodes[:-2]:
        #        i+=1
        #        if n.type=="Node":
        #            for nn in self.nodes[i:]:
        #                if nn.type=="Node":
        #                    self.Collide(n,nn)
            #for g in self.nodegroups[:-2]:
            #    for n in g.nodes:
            #        if n.type == "Nodegroup":
            #            for n in 
            for n in self.nodes:
                n.x+=n.pot[0]
                n.y+=n.pot[1]
        i=0
        for g in self.nodegroups:
            g.BoundingBoxSet(resize=True)
            g.pot=np.array([0.0,0.0])
        for g in self.nodegroups[:-2]:
            i+=1
            for gg in self.nodegroups[i:]:
                if (not(gg.parent==g)) and (not (g.parent==gg)):     
                    coll,dx,dy = self.Collide(g,gg)
                    if coll:
                        if abs(dx/g.sx)<abs(dy/g.sy):
                            #g.sx=g.sx-abs(dx)/2
                            g.x=g.x+dx/2
                            for n in g.nodes:
                                n.x=n.x+dx/2
                            #gg.sx=gg.sx-abs(dx)/2
                            gg.x=gg.x-dx/2
                            for n in gg.nodes:
                                n.x=n.x+dx/2

                        else:
                            #g.sy=g.sy-abs(dy)/2
                            g.y=g.y+dy/4
                            for n in g.nodes:
                                n.y=n.y+dy/2

                            #gg.sy=gg.sy-abs(dy)/2
                            gg.y=gg.y-dy/4
                            for n in gg.nodes:
                                n.y=n.y+dy/2
            

    def Collide(self,n1,n2, inside=False, potset=True):
        if max(np.abs(n1.pot))>90:
            #print("n1pot",n1.pot)
            n1.pot =np.array([0,0])
            return False, 0,0
        if max(np.abs(n2.pot))>90: 
            #print("n2pot",n2.pot)
            n2.pot =np.array([0,0])
            return False,0,0
        
        collx= False
        x1=n1.x+n1.pot[0]
        x2=n2.x+n2.pot[0]
        y1=n1.y+n1.pot[1]
        y2=n2.y+n2.pot[1]
        
        if x2>x1:
            dx=(x2-n2.sx)-(x1+n1.sx)
            if dx < 0: 
                collx=True
            tx=((n1.x+n1.sx)-(n2.x-n2.sx))/(n2.pot[0]-n1.pot[0])
        else:
            dx=(x2+n2.sx)-(x1-n1.sx)
            if dx > 0: 
                collx = True
            tx=((n1.x-n1.sx)-(n2.x+n2.sx))/(n2.pot[0]-n1.pot[0])

        colly= False
        if y2>y1:
            dy=(y2-n2.sy)-(y1+n1.sy)
            if dy < 0: 
                colly=True
            ty=((n1.y-n1.sy)-(n2.y+n2.sy))/(n2.pot[1]-n1.pot[1])
        else:
            dy=(y2+n2.sy)-(y1-n1.sy)
            if dy > 0: 
                colly = True
            ty=((n1.y+n1.sy)-(n2.y-n2.sy))/(n2.pot[1]-n1.pot[1])
        
        if inside:
            collx = not collx
            colly = not colly
        
        if collx and colly and potset:
            if np.isnan(tx): 
                print("TX",tx)
                tx=10
            if np.isnan(ty): 
                print("TY",ty)
                ty=10
            t=min(tx,ty)
            n1.pot=n1.pot*t
            n2.pot=n2.pot*t
        
        return (collx and colly), dx, dy
    
    def collapseNodegroups(self):
        for g in self.nodegroups:
            xc=0
            yc=0
            g.BoundingBoxSet(resize=True)
            g.updateConn()
            for n in g.conn:
                xc+=n.x
                yc+=n.y
            vec=np.array([g.x-xc,g.y-yc])    
            g.collapse(np.array(vec))
            self.MovenodesWithGroups()
    
    def MovenodesWithGroups(self):
        for g in self.nodegroups:
            BB=g.BoundingBoxSet(resize=False)
            x=BB[0]/2+BB[1]/2
            y=BB[2]/2+BB[3]/2
            dx=g.x-x
            dy=g.y-y
            for n in g.nodes:
                n.x=n.x+dx
                n.y=n.y+dy
            

    def collapseGraph(self):
        self.collapseNodegroups()
        self.MovenodesWithGroups()
        #self.BoundingBoxSet()
        #x=self.BB[0]/2+self.BB[1]/2
        #y=self.BB[2]/2+self.BB[3]/2
        #sx=self.BB[1]/2-self.BB[0]/2
        #sy=self.BB[3]/2-self.BB[2]/2
        #groupofnodegroups=nodegroup(x,y,sx,sy,self.nodegroups)
        #groupofnodegroups.collapse(np.array([-1,-1]))
        #self.MovenodesWithGroups()
    
    def MoveAll(self, ngs=True, ns=True, gps_in = False, sc=1, sc2=1):
        self.ClearPot()
        if ns:
            self.PotNodePoints(sc=sc,sc2=sc2)
        if ngs:
            self.PotNodegroupsPoints()
        #self.PotNodePoints()
        self.StepAllPot(ngs=ngs, ns=ns, gps_in=gps_in)

     
