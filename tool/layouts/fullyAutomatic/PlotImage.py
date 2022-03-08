import streamlit as st
import cv2 as cv
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from distinctipy import distinctipy

class FullyAutomaticRegionImage:

    def __init__(self,path,data,outputs_locked,outputMasks):
        self.path = path
        self.data = data
        self.outputsLocked = outputs_locked
        self.outputMasks = outputMasks
    
    def processData(self):
        self.image = cv.imread(self.path)

        self.colorMap = {}
        n = len(list(self.outputMasks.keys()))

        colors = distinctipy.get_colors(n)
        
        c = 0
        for k in self.outputMasks.keys():
            tempcolor = colors[c]
            self.colorMap[k] = {"polygon":"rgb("+str(tempcolor[0]*255)+","+str(tempcolor[1]*255)+","+str(tempcolor[2]*255)+")",
            "mask":"rgba("+str(tempcolor[0]*255)+","+str(tempcolor[1]*255)+","+str(tempcolor[2]*255)+",0.6)"}
            c += 1

    def getBbox(self):

        stacked_pts = self.data[list(self.outputMasks.keys())[0]]
        for out in list(self.outputMasks.keys())[1:]:
            stacked_pts = np.vstack((stacked_pts,self.data[out]))

        minx,miny = np.amin(stacked_pts,axis=0)
        maxx,maxy = np.amax(stacked_pts,axis=0)
        
        minx = max(0,minx-20)
        miny = max(0,miny-20)

        maxx = min(self.image.shape[1],maxx+20)
        maxy = min(self.image.shape[0],maxy+20)
        
        self.bbox = [int(minx),int(miny),int(maxx),int(maxy)]

    def renderImage(self):
        
        self.processData()
        self.getBbox()

        self.image = cv.cvtColor(self.image,cv.COLOR_RGB2BGR)
        self.image = self.image[self.bbox[1]:self.bbox[3],self.bbox[0]:self.bbox[2],:]

        pil_image = Image.fromarray(self.image)

        fig = go.Figure()
        actual_height, actual_width, _ = self.image.shape

        
        img_width = min(max(self.image.shape[1]/1.5,1000),1500)

        if actual_width/actual_height > 7:
            img_width = 1500
            img_height = 200
            
        elif actual_height/actual_width > 5:
            if actual_width < 200:
                img_width = 200
                img_height = 600
            else:
                img_height = 600
                img_width = 400
            
        else:
            if actual_width < 400 and actual_height < 400:
                img_width = 600
                img_height = actual_height*(600/actual_width)

        scale_factor = 1

        firstRegion = 0

        for out in self.outputMasks.keys():
        
            pts = np.array(self.data[out])

            pts[:,0] -= self.bbox[0]
            pts[:,1] -= self.bbox[1]

            xZoom = (np.take(pts,0,axis=1) * img_width * scale_factor / actual_width)
            yZoom = img_height * scale_factor - (np.take(pts,1,axis=1) * img_height * scale_factor / actual_height)

            if self.outputsLocked[out+"-polygon"]:
                fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,line_color=self.colorMap[out]["polygon"],name=out+"-polygon"))
            else:
                fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,line_color=self.colorMap[out]["polygon"],name=out+"-polygon",visible="legendonly"))
            
            if self.outputsLocked[out+"-pts"]:
                fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,name=out+"-pts",mode='markers',marker_color=self.colorMap[out]["polygon"]))    
            else:
                fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,name=out+"-pts",mode='markers',marker_color=self.colorMap[out]["polygon"],visible="legendonly"))
            
            if self.outputMasks[out+""]:
                if self.outputsLocked[out+"-mask"]:
                    fig.add_trace(go.Scatter(x = xZoom,y = yZoom,fill="toself",mode='none',name=out+"-mask",fillcolor=self.colorMap[out]["mask"],opacity=0.3))    
                else:
                    fig.add_trace(go.Scatter(x = xZoom,y = yZoom,fill="toself",mode='none',name=out+"-mask",fillcolor=self.colorMap[out]["mask"],opacity=0.3,visible='legendonly'))    
                
        
        xAxisRange = img_width * scale_factor
        yAxisRange = img_height * scale_factor

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, xAxisRange]
        )
        
        fig.update_yaxes(
            visible=False,
            range=[0, yAxisRange],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=pil_image)
        )
        # Configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        fig.update_layout(legend=dict(
            orientation="h",
            # xanchor="right",
            yanchor="bottom",
            y=-0.1, 
            x=0   
        ))
        
        return fig

class FullDocumentImage:

    def __init__(self,data,outputs_locked,outputMasks,regionShown=None):
        self.data = data
        self.path = data["imagePath"]
        self.regionShown = regionShown
        self.outputsLocked = outputs_locked
        self.outputMasks = outputMasks
        # self.outputs = data["outputs"]
    
    def processData(self):
        self.image = cv.imread(self.path)

        self.colorMap = {}
        n = len(list(self.outputMasks.keys()))

        colors = distinctipy.get_colors(n)
        
        c = 0
        for k in self.outputMasks.keys():
            tempcolor = colors[c]
            self.colorMap[k] = {"polygon":"rgb("+str(tempcolor[0]*255)+","+str(tempcolor[1]*255)+","+str(tempcolor[2]*255)+")",
            "mask":"rgba("+str(tempcolor[0]*255)+","+str(tempcolor[1]*255)+","+str(tempcolor[2]*255)+",0.6)"}
            c += 1
    
    def renderImage(self):
        self.processData()

        self.image = cv.cvtColor(self.image,cv.COLOR_RGB2BGR)
        pil_image = Image.fromarray(self.image)

        fig = go.Figure()
        actual_height, actual_width, _ = self.image.shape

        
        img_width = min(max(self.image.shape[1]/1.5,1000),1500)

        if actual_height > 800:
            img_height = min(max(self.image.shape[0],800),900)
        else:
            img_height = min(max(self.image.shape[0],500),600)

        # img_width = actual_width
        # img_height = actual_height

        # print(img_height/actual_height)
        # print(img_width/actual_width)

        scale_factor = 1

        for region in self.data["regions"]:

            cnt = 0

            for out in self.outputMasks.keys():
                pts = np.array(region[out])
                pts = np.concatenate((pts,[pts[0]]),axis=0)

                xZoom = (np.take(pts,0,axis=1) * img_width * scale_factor / actual_width)
                yZoom = img_height * scale_factor - (np.take(pts,1,axis=1) * img_height * scale_factor / actual_height)

                if self.outputsLocked[out+"-polygon"]:
                    fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,line_color=self.colorMap[out]["polygon"],name=out+"-polygon",legendgroup=str(cnt+1),showlegend=(region==self.data["regions"][0])))
                else:
                    fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,line_color=self.colorMap[out]["polygon"],name=out+"-polygon",visible='legendonly',legendgroup=str(cnt+1),showlegend=(region==self.data["regions"][0])))
                
                if self.outputsLocked[out+"-pts"]:
                    fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,name=out+"-pts",mode='markers',marker_color=self.colorMap[out]["polygon"],legendgroup=str(cnt+2),showlegend=(region==self.data["regions"][0])))    
                else:
                    fig.add_trace(go.Scattergl(x = xZoom,y = yZoom,name=out+"-pts",visible='legendonly',mode='markers',marker_color=self.colorMap[out]["polygon"],legendgroup=str(cnt+2),showlegend=(region==self.data["regions"][0])))    
                
                if self.outputMasks[out]:
                    if self.outputsLocked[out+"-mask"]:
                        fig.add_trace(go.Scatter(x = xZoom,y = yZoom,fill="toself",mode='none',name=out+"-mask",fillcolor=self.colorMap[out]["mask"],legendgroup=str(cnt+3),showlegend=(region==self.data["regions"][0])))    
                    else:
                        fig.add_trace(go.Scatter(x = xZoom,y = yZoom,fill="toself",mode='none',name=out+"-mask",fillcolor=self.colorMap[out]["mask"],visible='legendonly',legendgroup=str(cnt+3),showlegend=(region==self.data["regions"][0])))    

                cnt += 3
        
        xAxisRange = img_width * scale_factor
        yAxisRange = img_height * scale_factor

        # Configure axes
        fig.update_xaxes(
            visible=False,
            range=[0, xAxisRange]
        )
        
        fig.update_yaxes(
            visible=False,
            range=[0, yAxisRange],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x"
        )

        # Add image
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=pil_image)
        )
        # Configure other layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        fig.update_layout(legend=dict(
            orientation="h",
            # xanchor="right",
            yanchor="bottom",
            y=-0.1, 
            x=0   
        ))
        
        return fig
