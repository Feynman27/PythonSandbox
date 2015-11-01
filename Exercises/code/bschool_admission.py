#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze(data,schoolname):
# those invited to interview
    data_invited = data[(data['Round']=='R1')&(data['Status']=='Invited to Interview')]
    data_invited=data_invited.reset_index()
# those not invited
    data_ninvited = data[(data['Round']=='R1')&(data['Status']!='Invited to Interview')]
    data_ninvited=data_ninvited.reset_index()

    gmat_invited_average = data_invited[['GMAT']].astype(float).mean()
    gpa_invited_average = data_invited[['UG GPA']].astype(float).mean()
    tot_r1_apps = sum(data[(data['Round']=='R1')]['Username'].value_counts(dropna=False))

    print '====='+schoolname+ '====='
    print '# invited to interview in R1: ' + str(sum(data_invited['Username'].value_counts(dropna=False)))
    print '[%] invited to interview: ' + str(sum(data_invited['Username'].value_counts(dropna=False)).astype(float)/tot_r1_apps.astype(float)*100.0)+('[%]')
    print 'Average GMAT: ' + str(gmat_invited_average )
    print 'Response [%] ' + str(data_invited['GMAT'].count().astype(float)/data_invited['Username'].count().astype(float)*100.0)
    print 'Average GPA: ' + str(gpa_invited_average)
    print 'Response [%] ' + str(data_invited['UG GPA'].count().astype(float)/data_invited['Username'].count().astype(float)*100.0)
    print '========================'

    gmat_ninvited_average = data_ninvited[['GMAT']].astype(float).mean()
    gpa_ninvited_average = data_ninvited[['UG GPA']].astype(float).mean()
    print '# not invited to interview in R1: ' + str(sum(data_ninvited['Username'].value_counts(dropna=False)))
    print '[%] not invited to interview: ' + str(sum(data_ninvited['Username'].value_counts(dropna=False)).astype(float)/tot_r1_apps.astype(float)*100.0)+('[%]')
    print 'Average GMAT: ' + str(gmat_ninvited_average )
    print 'Response [%] ' + str(data_ninvited['GMAT'].count().astype(float)/data_ninvited['Username'].count().astype(float)*100.0)
    print 'Average GPA: ' + str(gpa_ninvited_average)
    print 'Response [%] ' + str(data_ninvited['UG GPA'].count().astype(float)/data_ninvited['Username'].count().astype(float)*100.0)
    print '========================\n'

# gmat
    fig = plt.figure('GMAT '+schoolname)
    plt.title(schoolname)
    bins=np.arange(640,790+10,10)
    data_invited['GMAT'].plot(kind='hist',bins=bins,color='b',alpha=0.3,label='Invited',hatch='/')
    data_ninvited['GMAT'].plot(kind='hist',bins=bins,color='g',alpha=0.2,label='Not Invited')
#plt.hist(data_invited['GMAT'].value_counts(),data_invited['GMAT'].nunique(),facecolor='green',alpha=0.5)
    plt.ylabel('GMAT Club Members')
    plt.xlabel('GMAT')
    _=plt.xlim([650,790])
    _=plt.ylim([0,20])
    plt.text(660, 15, r'$\mu_{not\/invited}=' + str(np.round(np.mean(gmat_ninvited_average),2)) + ' $',fontsize=14)
    plt.text(660, 13, r'$\mu_{invited}=' + str(np.round(np.mean(gmat_invited_average),2)) + ' $',fontsize=14)
    plt.legend()

# gpa
    fig = plt.figure('GPA '+schoolname)
    plt.title(schoolname)
    bins=np.arange(2.5,4.1+0.1,0.1)
    data_invited['UG GPA'].plot(kind='hist',bins=bins,color='b',alpha=0.3,label='Invited',hatch='/')
    data_ninvited['UG GPA'].plot(kind='hist',bins=bins,color='g',label='Not Invited',alpha=0.2)
    plt.ylabel('GMAT Club Members')
    plt.xlabel('UG GPA')
    _=plt.xlim([2.5,4.1])
    _=plt.ylim([0,15])
    plt.text(2.6, 12, r'$\mu_{not\/invited}=' + str(np.round(np.mean(gpa_ninvited_average),2)) + ' $',fontsize=14)
    plt.text(2.6, 10, r'$\mu_{invited}=' + str(np.round(np.mean(gpa_invited_average),2)) + ' $',fontsize=14)
    plt.legend()

# plot 2D scatter
    fig = plt.figure('gpa vs. gmat '+schoolname,figsize=(8,8))
    from matplotlib.ticker import NullFormatter
# x,y must be numpy.ndarray
    x = data_invited['GMAT'].values
    y = data_invited['UG GPA'].values

    nullfmt   = NullFormatter() # no labels

# size, height, and width of the scatter and projections
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.set_title(schoolname)
    axScatter.xaxis.set_label_text('GMAT')
    axScatter.yaxis.set_label_text('UG GPA')
    axHistx.yaxis.set_label_text('Invitees')
    axHisty.xaxis.set_label_text('Invitees')

# eliminate x,y labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
# plot scatter
    axScatter.scatter(x, y)

    axScatter.set_xlim( (650, 790) )
    axScatter.set_ylim( (2.5, 4.1) )

    binsx=np.arange(650,790+10,10)
    binsy=np.arange(2.5, 4.1+0.1,0.1)

    axHistx.hist(x, bins=binsx)
    axHisty.hist(y,bins=binsy,orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

# create map of country --> (#invited,gmat,gpa)
    #data_cntries = data_invited.groupby('Country')['Country'].count()
    data_cntries = pd.concat([data_invited.groupby('Country')['Country'].count(), data_ninvited.groupby('Country')['Country'].count()],axis=1)
    data_cntries.columns=['Invited','NotInvited']
# fill nan with zeros
    data_cntries = data_cntries.fillna(0)

# Calculate average GMAT and GPA per country
# list of (gmat,gpa) tuples
    average_gmat_gpa_inv={}
    average_gmat_gpa_ninv={}
    for cntry in data_cntries.index:
        gmat_sum_inv = 0
        gpa_sum_inv  = 0

        gmat_sum_ninv = 0
        gpa_sum_ninv  = 0

        # total number of invitees in country
        tot_gmat_cntry_inv = data_cntries['Invited'][cntry].astype(float)
        tot_gmat_cntry_ninv = data_cntries['NotInvited'][cntry].astype(float)

        # gpa,gmat columns may have different
        # number of nan values
        tot_gpa_cntry_inv = tot_gmat_cntry_inv
        tot_gpa_cntry_ninv = tot_gmat_cntry_ninv

        # iterate over invitees
        for inv,inv_cntry in enumerate(data_invited['Country']):
            # this is the current country of interest
            if (inv_cntry==cntry):
                if not np.isnan(data_invited['GMAT'][inv]) == True:
                    gmat_sum_inv += data_invited['GMAT'][inv]
                # if no entry, remove data point
                else : tot_gmat_cntry_inv-=1
                if not  np.isnan(data_invited['UG GPA'][inv]) == True:
                    gpa_sum_inv += data_invited['UG GPA'][inv]
                else : tot_gpa_cntry_inv-=1

        # map for invitees
        average_gmat_gpa_inv[cntry] = [data_cntries['Invited'][cntry],gmat_sum_inv/tot_gmat_cntry_inv,gpa_sum_inv/tot_gpa_cntry_inv]

        # iterate over non-invitees
        for ninv,ninv_cntry in enumerate(data_ninvited['Country']):
            # this is the current country of interest
            if (ninv_cntry==cntry):
                if not np.isnan(data_ninvited['GMAT'][ninv]) == True:
                    gmat_sum_ninv += data_ninvited['GMAT'][ninv]
                # if no entry, remove data point
                else : tot_gmat_cntry_ninv-=1
                if not  np.isnan(data_ninvited['UG GPA'][ninv]) == True:
                    gpa_sum_ninv += data_ninvited['UG GPA'][ninv]
                else : tot_gpa_cntry_ninv-=1

        # map for ninvitees
        average_gmat_gpa_ninv[cntry] = [data_cntries['NotInvited'][cntry],gmat_sum_ninv/tot_gmat_cntry_ninv,gpa_sum_ninv/tot_gpa_cntry_ninv]

#convert to dataframe
    data_country = pd.DataFrame.from_dict(average_gmat_gpa_inv,orient='index')
    data_country.columns=['Invitees','Avg Inv GMAT','Avg Inv GPA']
    data_country_ninv = pd.DataFrame.from_dict(average_gmat_gpa_ninv,orient='index')
    data_country_ninv.columns=['Not Invited','Avg nInv GMAT','Avg nInv GPA']
# merge inv and non-invitees (sorted by country)
    data_country=data_country.join(data_country_ninv)
    data_country[['Invitees','Not Invited']]=data_country[['Invitees','Not Invited']].fillna(0)
    print data_country
    #data_country['Avg GPA'] = data_country['Avg GPA'].map(lambda x: '%2.2f' % x)
    #data_country['Avg GMAT'] = data_country['Avg GMAT'].map(lambda x: '%2.1f' % x)

# country of origin
    #fig = plt.figure('Country of Origin ' +schoolname)
    fig, ax = plt.subplots(114)
    plt.title(schoolname)
    #ax=data_country['Invitees'].plot(kind='bar',color='r')
    ax=data_country[['Invitees','Not Invited']].plot(kind='bar',stacked=True)
    #inv_hts, inv_bins = np.histogram(data_country['Invitees'])
    #ninv_hts, ninv_bins = np.histogram(data_country['Not Invited'],bins=inv_bins)

    #width = (inv_bins[1]-inv_bins[0])/3

    #ax.bar(inv_bins[:-1],inv_hts,width=width,facecolor='cornflowerblue')
    #ax.bar(ninv_bins[:-1]+width,ninv_hts,width=width,facecolor='seagreen')

    ax.set_xticklabels(data_country.index,rotation=40,fontsize=8,ha='right')
    ax.set_xlabel('Country')
    ax.set_ylabel('GMAT Club Members')
    ax.set_ylim((0,50))
    for i,cntry in enumerate(data_country.index):
        gmat = '%2.1f' % data_country['Avg Inv GMAT'][cntry]
        gpa = '%3.1f' % data_country['Avg Inv GPA'][cntry]
        if(np.isnan(data_country['Avg Inv GMAT'][cntry])==True | np.isnan(data_country['Avg Inv GPA'][cntry])==True) : continue
        if (i==0) :
            annot_gmat = 'Avg GMAT: ' + str(gmat)
            annot_gpa = 'Avg GPA: ' + str(gpa)
            xoffset = -0.2
        else :
            annot_gmat=str(gmat)
            annot_gpa=str(gpa)
            xoffset=0.0

        yoffset = 2.5

        # GMAT
        ax.annotate(annot_gmat,xy=(i+xoffset,data_country['Invitees'][cntry]+yoffset))
        # GPA
        ax.annotate(annot_gpa,xy=(i+xoffset,data_country['Invitees'][cntry]+yoffset+2.4))

    plt.show()

if __name__ == '__main__':
    data_ross = pd.read_csv('../../data/Ross_Michigan_applicants_Cof2018.csv',na_values=['','-'])
    analyze(data_ross,"Ross")
