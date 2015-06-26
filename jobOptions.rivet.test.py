## Example job option script to run an event generator
## and Rivet inside Athena
##
## Author: James Monk <jmonk@hep.ucl.ac.uk>
## Author: Andy Buckley <andy.buckley@cern.ch>

include("GeneratorUtils/StdEvgenSetup.py")
theApp.EvtMax = 1000

## Configure and add an event generator to the alg seq
from Pythia8_i.Pythia8_iConf import Pythia8_i
topAlg += Pythia8_i("Pythia8")
topAlg.Pythia8.CollisionEnergy = 7000.0
topAlg.Pythia8.Commands += ['HardQCD:all = on', 'PhaseSpace:pTHatMin = 30.0']

## Retrieve keywords in metadata_Generate.xml (from Generate_tf.py)
keywords = []
import xml.etree.ElementTree as ET
tree = ET.parse('metadata_Generate.xml')
root = tree.getroot()
for META in root.findall('META'):
  name = META.get('name')
  if name == 'keywords':
    keywords = META.get('value')
    keywords = keywords.split(",")
    keywords = [x.strip(' ') for x in keywords]
if len(keywords) > 0:
    print '==================================================================================================================================='
    print 'Found keywords: '+str(keywords)
    print '==================================================================================================================================='
else:
  print "Cannot locate keywords in metadata."
  exit()

########################################################
## Mapping Rivet analyses --> MC14 keywords
########################################################
analysisSet = set()
## Jets
if any(k in keywords for k in ['jets','monojet','1jet','5jet']):
  analysisSet.add('MC_LEADJETUE')
  analysisSet.add('MC_QCD_PARTONS')
  analysisSet.add('ALEPH_2004_S5765862')
  analysisSet.add('ATLAS_2010_CONF_2010_049')
  analysisSet.add('ATLAS_2011_I919017')
  analysisSet.add('ATLAS_2011_S8924791')
  analysisSet.add('ATLAS_2012_I1094564')
  analysisSet.add('ATLAS_2012_I1119557')
  analysisSet.add('CDF_2001_S4563131')
  analysisSet.add('CDF_2005_S6217184')
  analysisSet.add('CDF_2006_S6450792')
  analysisSet.add('CDF_2007_S7057202')
  analysisSet.add('CDF_2008_S7828950')
  analysisSet.add('CMS_2011_S9120041')
  analysisSet.add('CMS_2012_I1087342')
  analysisSet.add('CMS_2013_I1224539_DIJET')
  analysisSet.add('D0_1996_S3324664')
  analysisSet.add('D0_2004_S5992206')
  analysisSet.add('D0_2008_S7662670')
  analysisSet.add('H1_1995_S3167097')
  analysisSet.add('JADE_OPAL_2000_S4300807')
  analysisSet.add('STAR_2006_S6870392')
if 'dijet' in keywords:
  analysisSet.add('CDF_2008_S8093652')
  analysisSet.add('ATLAS_2011_I930220')
  analysisSet.add('ATLAS_2011_S8971293')
  analysisSet.add('ATLAS_2011_S9126244')
  analysisSet.add('ATLAS_2012_I1082936')
  analysisSet.add('ATLAS_2012_I1188891')
  analysisSet.add('ATLAS_2010_S8817804')
  analysisSet.add('CDF_1996_S3418421')
  analysisSet.add('CDF_2000_S4266730')
  analysisSet.add('CDF_2008_S8093652')
  analysisSet.add('CMS_2011_S8950903')
  analysisSet.add('CMS_2011_S8968497')
  analysisSet.add('CMS_2011_S9215166')
  analysisSet.add('CMS_2012_I1102908')
  analysisSet.add('CMS_2012_I1184941')
  analysisSet.add('CMS_2013_I1224539_DIJET')
  analysisSet.add('D0_2009_S8320160')
  analysisSet.add('D0_2010_S8566488')
  analysisSet.add('MC_DIJET')
  analysisSet.add('ZEUS_2001_S4815815')
if '2jet' in keywords:
  analysisSet.add('CDF_2001_S4517016')
if '3jet' in keywords:
  analysisSet.add('D0_2011_I895662')
if all(k in keywords for k in ['2jet','3jet']):
  analysisSet.add('CMS_2011_S9088458')
  analysisSet.add('D0_2011_I895662')
if '4jet' in keywords:
  analysisSet.add('OPAL_2001_S4553896')
  analysisSet.add('DELPHI_2003_WUD_03_11')
  analysisSet.add('CMS_2013_I1273574')
if all(k in keywords for k in ['3jet','4jet']):
  analysisSet.add('D0_1996_S3214044')
if '6jet' in keywords:
  analysisSet.add('CDF_1997_S3541940')
if 'multijet' in keywords:
  analysisSet.add('ATLAS_2011_S9128077')
  analysisSet.add('CDF_1996_S3108457')
  analysisSet.add('CDF_1996_S3349578')
## Photon
if any(k in keywords for k in ['photon','monophoton','1photon','3photon','4photon']):
  analysisSet.add('MC_PHOTONS')
  analysisSet.add('MC_PHOTONINC')
  analysisSet.add('ATLAS_2010_S8914702')
  analysisSet.add('CDF_1993_S2742446')
  analysisSet.add('CDF_2009_S8436959')
  analysisSet.add('D0_2006_S6438750')
  analysisSet.add('OPAL_1993_S2692198')
if all(k in keywords for k in ['photon','quark']:
  analysisSet.add('ALEPH_1996_S3196992')
if any(k in keywords for k in ['diphoton','2photon']:
  analysisSet.add('MC_DIPHOTON')
  analysisSet.add('ATLAS_2011_S9120807')
  analysisSet.add('CDF_2005_S6080774')
  analysisSet.add('D0_2010_S8570965')
if all(k in keywords for k in ['photon','muon']:
  analysisSet.add('CMS_2011_I954992')
if all(k in keywords for k in ['photon','jets']:
  analysisSet.add('MC_PHOTONJETS')
  analysisSet.add('MC_PHOTONJETUE')
  analysisSet.add('MC_PHOTONKTSPLITTINGS')
  analysisSet.add('ATLAS_2012_I1093738')
  analysisSet.add('D0_2008_S7719523')
if all(k in keywords for k in ['diphoton','SUSY']:
  analysisSet.add('ATLAS_2012_I946427')
if all(k in keywords for k in ['photon','Z']:
  analysisSet.add('OPAL_1998_S3749908')
## Higgs
if 'Higgs' in keywords:
  analysisSet.add('MC_HINC')
if all(k in keywords for k in ['Higgs','jets']
  analysisSet.add('MC_HJETS')
  analysisSet.add('MC_HKTSPLITTINGS')
## Identified particles
if 'singleParticle' in keywords:
  analysisSet.add('MC_IDENTIFIED')
## SUSY
if 'SUSY' in keywords:
  analysisSet.add('MC_SUSY')
if all(k in keywords for k in ['SUSY','1lepton']:
  analysisSet.add('ATLAS_2011_CONF_2011_090')
  analysisSet.add('ATLAS_2011_S9212353')
if all(k in keywords for k in ['SUSY','2lepton']:
  analysisSet.add('ATLAS_2011_S9019561')
  analysisSet.add('ATLAS_2012_I943401')
if all(k in keywords for k in ['SUSY','jets','1lepton']:
  analysisSet.add('ATLAS_2012_CONF_2012_104')
if all(k in keywords for k in ['SUSY','jets','2lepton','sameSign']:
  analysisSet.add('ATLAS_2012_CONF_2012_105')
if all(k in keywords for k in ['SUSY','jets','bottom','lepton']:
  analysisSet.add('ATLAS_2012_I1095236')
if all(k in keywords for k in ['SUSY','jets','bottom']:
  analysisSet.add('ATLAS_2011_CONF_2011_098')
if all(k in keywords for k in ['SUSY','top']:
  analysisSet.add('ATLAS_2012_I1126136')
if all(k in keywords for k in ['SUSY','3lepton']:
  analysisSet.add('ATLAS_2012_I1112263')
if all(k in keywords for k in ['SUSY','jets','lepton']:
  analysisSet.add('ATLAS_2012_I1180197')
if all(k in keywords for k in ['SUSY','4lepton']:
  analysisSet.add('ATLAS_2012_CONF_2012_001')
  analysisSet.add('ATLAS_2012_CONF_2012_153')
  analysisSet.add('ATLAS_2012_I1190891')
## ttbar
if 'ttbar' in keywords:
  analysisSet.add('MC_TTBAR')
  analysisSet.add('ATLAS_2012_I1094568')
## VH2BB
if all(k in keywords for k in ['bottom','bottomonium','Higgs']):
  analysisSet.add('MC_VH2BB')
## Z
if 'Z' in keywords:
  analysisSet.add('ATLAS_2011_S9131140')
  analysisSet.add('ATLAS_2012_I1204784')
if all(k in keywords for k in ['Z','electron']):
  analysisSet.add('MC_ZINC')
  analysisSet.add('CDF_2000_S4155203')
  analysisSet.add('CDF_2009_S8383952')
if all(k in keywords for k in ['Z','muon']):
  analysisSet.add('CMS_2012_I941555')
if all(k in keywords for k in ['Z','drellYan']):
  analysisSet.add('CDF_2010_S8591881_DY')
if all(k in keywords for k in ['Z','allHadronic']):
  analysisSet.add('ALEPH_1991_S2435284')
  analysisSet.add('ALEPH_2002_S4823664')
  analysisSet.add('DELPHI_1995_S3137023')
  analysisSet.add('DELPHI_1999_S3960137')
  analysisSet.add('JADE_1998_S3612880')
  analysisSet.add('OPAL_1995_S3198391')
  analysisSet.add('OPAL_1997_S3608263')
  analysisSet.add('OPAL_1998_S3702294')
  analysisSet.add('OPAL_1998_S3749908')
  analysisSet.add('OPAL_2000_S4418603')
if all(k in keywords for k in ['Z','bottom','charm']):
  analysisSet.add('OPAL_1998_S3780481')
if all(k in keywords for k in ['Z','Lambda']):
  analysisSet.add('OPAL_1997_S3396100')
if all(k in keywords for k in ['Z','Jpsi']):
  analysisSet.add('OPAL_1996_S3257789')
if all(k in keywords for k in ['Z','Kplus','Kminus']):
  analysisSet.add('OPAL_1994_S2927284')
if all(k in keywords for k in ['Z','bottomonium','charmonium']):
  analysisSet.add('OPAL_2002_S5361494')
  analysisSet.add('SLD_1996_S3398250')
  analysisSet.add('DELPHI_2000_S4328825')
if all(k in keywords for k in ['Z','Lambda','Kplus','Kminus']):
  analysisSet.add('SLD_1999_S3743934')
if all(k in keywords for k in ['Z','Lambda','Kplus','Kminus','bottom','charm']):
  analysisSet.add('SLD_2004_S5693039')
if all(k in keywords for k in ['Z','bottom']):
  analysisSet.add('ALEPH_2001_S4656318')
  analysisSet.add('SLD_2002_S4869273')
  analysisSet.add('DELPHI_2002_069_CONF_603')
if all(k in keywords for k in ['Z','bottom','jets']):
  analysisSet.add('CDF_2006_S6653332')
  analysisSet.add('CDF_2008_S8095620')
if all(k in keywords for k in ['Z','jets']):
  analysisSet.add('ATLAS_2011_I945498')
  analysisSet.add('ATLAS_2013_I1230812')
  analysisSet.add('CDF_2008_S7540469')
  analysisSet.add('CMS_2013_I1209721')
  analysisSet.add('D0_2010_S8821313')
if all(k in keywords for k in ['Z','jets','muon']):
  analysisSet.add('ATLAS_2013_I1230812_MU')
  analysisSet.add('D0_2008_S7863608')
  analysisSet.add('D0_2009_S8349509')
  analysisSet.add('D0_2010_S8671338')
if all(k in keywords for k in ['Z','jets','electron']):
  analysisSet.add('MC_ZJETS')
  analysisSet.add('MC_ZKTSPLITTINGS')
  analysisSet.add('ATLAS_2013_I1230812_EL')
  analysisSet.add('CMS_2013_I1224539_ZJET')
  analysisSet.add('D0_2007_S7075677')
  analysisSet.add('D0_2008_S6879055')
  analysisSet.add('D0_2008_S7554427')
  analysisSet.add('D0_2009_S8202443')
if all(k in keywords for k in ['Z','jets','photon']):
  analysisSet.add('CMS_2013_I1258128')
## ZZ
if all(k in keywords for k in ['diboson','ZZ']):
  analysisSet.add('MC_ZZINC')  
if all(k in keywords for k in ['diboson','ZZ','jets']):
  analysisSet.add('MC_ZZJETS')
  analysisSet.add('MC_ZZKTSPLITTINGS')
## VHiggs
if 'bottomonium' in keywords and any(k in keywords for k in ['ZHiggs','WHiggs']):
  analysisSet.add('MC_VH2BB')
## W
if 'W' in keywords:
  analysisSet.add('ATLAS_2011_I925932')
if all(k in keywords for k in ['W','jets']):
  analysisSet.add('ATLAS_2010_S8919674')
  analysisSet.add('ATLAS_2012_I1083318')
  analysisSet.add('ATLAS_2013_I1217867')
if all(k in keywords for k in ['W','electron']):
  analysisSet.add('MC_WINC')
  analysisSet.add('MC_WKTSPLITTINGS')
  analysisSet.add('MC_WPOL')
  analysisSet.add('D0_2000_S4480767')
  analysisSet.add('D0_2008_S7837160')
if all(k in keywords for k in ['W','muon']):
  analysisSet.add('ATLAS_2011_S9002537')
if all(k in keywords for k in ['W','jets','electron']):
  analysisSet.add('MC_WJETS')
  analysisSet.add('CDF_2008_S7541902')
  analysisSet.add('CMS_2013_I1224539_WJET')
if all(k in keywords for k in ['W','2jet','muon']):
  analysisSet.add('CMS_2013_I1272853')
## WW
if all(k in keywords for k in ['diboson','WW']):
  analysisSet.add('MC_WWINC')  
if all(k in keywords for k in ['diboson','WW','jets']):
  analysisSet.add('MC_WWJETS')
  analysisSet.add('MC_WWKTSPLITTINGS')
#WZ
if all(k in keywords for k in ['diboson','WZ']):
  analysisSet.add('ATLAS_2011_I954993')
if all(k in keywords for k in ['diboson','WZ','electron']):
  analysisSet.add('D0_2001_S4674421')
## Cross-section
if any(k in keywords for k in ['minBias','NLO']): 
  analysisSet.add('MC_XS')

print 'Found keyword matching Rivet analysis : ' + str(analysisSet)
########################################################

## Now set up the appropriate Rivet analyses
from Rivet_i.Rivet_iConf import Rivet_i
topAlg += Rivet_i("Rivet")
topAlg.Rivet.Analyses = ["MC_GENERIC","MC_JETS","MC_PDFS"]
## Now add Rivet analyses to list of analyses to run
for iAn in analysisSet:
  topAlg.Rivet.Analyses += [iAn]
#topAlg.Rivet.DoRootHistos = False
#topAlg.Rivet.OutputLevel = DEBUG

## Configure ROOT file output
from AthenaCommon.AppMgr import ServiceMgr as svcMgr
from GaudiSvc.GaudiSvcConf import THistSvc
svcMgr += THistSvc()
svcMgr.THistSvc.Output = ["Rivet DATAFILE='Rivet.root' OPT='RECREATE'"]
#svcMgr.MessageSvc.OutputLevel = ERROR
