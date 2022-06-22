# Simplified COVID-19 epidemiological model & City mobility analysis

## Introduction

This repository contains the codes for the project of the courses [Physics of Complex systems](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2021/433619) and [Physical Methods of Biology](https://www.unibo.it/en/teaching/course-unit-catalogue/course-unit/2021/433617) for the curriculum of Applied Physics at the University of Bologna.

Files on the repository:
- ```model.py``` code for the simulation of a COVID-19 pandemic wave according to a distributed delay, six compartment SIR model;
- ```rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv``` dataset regarding the autoveichles mobility in the city of Bologna during 2020, offered by [Open Data Bologna](https://opendata.comune.bologna.it/pages/home/);
- ```mobility-analysis.py``` code used to analyze the dataset;
- ```mappa1080.png``` a 1517x1080 map of the city of Bologna;
- ```images``` folder containing some plots.

## Aim of the project

The aim of the project is to reconstruct the hospitalization dynamics during the first COVID-19 pandemic wave in the city of Bologna, starting from a six compartment SIR model. This can be done if we compare the **real** private data concerning the number of people that have been hospetalized in 2020 with the **modelled** data, that can be achieved by analyzing the autoveichles mobility in the city of Bologna during 2020 and after that, inserting some mobility's information into the SIR model.

## What we have done up to now

First of all, we coded a six compartment SIR model, containing:
- S, Susceptibles compartment;
- E, Exposed compartment;
- I, Infected compartment;
- H, Hospitalised compartment;
- A, Asymptomatic compartment;
- R, Removed compartment.

Of course in the equations we take into account the social activity rate. 
In the example it is reported a a quite strong lockdown 30 days after the introduction of patient zero:
<p align="center">
  <img src="https://github.com/keivan-amini/simplified-covid-model/blob/main/images/Figure_0-.png?raw=true" align="centre" height="400" width="600"  alt="SIR model"/>
</p>

In the analyzed dataset, we started figuring out the geographical position of the autoveichles detectors. In this image, every blue circle corresponds to an autoveichles detector.
<p align="center">
  <img src="https://github.com/keivan-amini/simplified-covid-model/blob/main/images/Figure_1.png?raw=true" align="centre" height="500" width="1500"  alt="map"/>
</p>

We also tried to figure out which streets were more busy. In order to do that, as an example we just focused on the January 2020 dataframe and we performed a scatter plot with the size of the points related to an *Average Monthly Mobility* coefficient. To better understand it, I suggest to look at the written code ```mobility-analysis.py```. However, the result is the following.
<p align="center">
  <img src="https://github.com/keivan-amini/simplified-covid-model/blob/main/images/Figure_3.png?raw=true" align="centre" height="500" width="1500"  alt="map"/>
</p>

After that, we focused on an interesting street: the so-called *Tangenziale Casalecchio - San Lazzaro*
TODO
