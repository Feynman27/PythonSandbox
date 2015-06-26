#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#include "TROOT.h"
#include "TStyle.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TColor.h"
#include "TPaveText.h"
#include "TObjString.h"
#include "TControlBar.h"

#include "TGWindow.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"

#include "TMVA/DecisionTree.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TXMLEngine.h"

void decisionTree( )
{

    TFile* outputFile = TFile::Open( "TMVA.root", "RECREATE" );

    TMVA::Factory *factory = new TMVA::Factory( "MVAnalysis", outputFile,"!V:Transformations=I;N;D");

    TFile *input = new TFile("output.root","READ");
    TTree* t = (TTree*)input->Get("tree");

    factory->AddVariable("ConsExp_marital", 'F');
    factory->AddVariable("ConsExp_foodhome", 'F');   

    factory->SetInputTrees(t,"ConsExp_marital==1","ConsExp_marital!=1.0");

    factory->PrepareTrainingAndTestTree( "", "",
            "nTrain_Signal=200:nTrain_Background=200:nTest_Signal=200:nTest_Background=200:!V" );

    factory->BookMethod( TMVA::Types::kLikelihood, "Likelihood",
            "!V:NAvEvtPerBin=50" );

    factory->BookMethod( TMVA::Types::kMLP, "MLP", 
            "!V:NCycles=50:HiddenLayers=10,10:TestRate=5" );

    factory->BookMethod( TMVA::Types::kBDT, "BDT", 
            "!V:BoostType=Grad:nCuts=20:NNodesMax=5" );

    factory->TrainAllMethods();  
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    outputFile->Close();
    delete factory;
}
