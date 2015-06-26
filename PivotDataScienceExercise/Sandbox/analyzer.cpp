#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"
#include "TInterpreter.h"

void fillVector(int column_no, float value, std::map<std::string, std::vector<float>* >& m_Values, std::map<int, std::string>& m_Index){
    /*if(column_no==0) m_ConsExp["newid"]->push_back(value);
    if(column_no==1) m_ConsExp["age"]->push_back(value);
    if(column_no==2) m_ConsExp["educatio"]->push_back(value);
    if(column_no==3) m_ConsExp["race"]->push_back(value);
    if(column_no==4) m_ConsExp["sex"]->push_back(value);
    */
    // if(column_no==5) m_ConsExp[""]->push_back(value);
    // if(column_no==6) m_ConsExp[""]->push_back(value);
    // if(column_no==7) m_ConsExp["incoll"]->push_back(value);
    m_Values[ m_Index[column_no] ]->push_back(value);
}

int main()
{
    gInterpreter->EnableAutoLoading();
    //gInterpreter->GenerateDictionary("std::vector<float>", "vector.h;vector");

    std::cout << "Starting the analyzer!" << std::endl;
    /// Output file
    TFile* _fOut = new TFile("output.root","recreate");
    TTree* outputTree = new TTree("tree","tree");
    std::map<std::string, std::vector<float>* > m_ConsExp;
    std::map<int, std::string> m_ConsExpIndex;
    std::map<std::string, std::vector<float>* > m_Prices;
    std::map<int, std::string> m_PricesIndex;
    
    //std::string input = "local_file.csv";
    std::string input = "Data_consumer_expenditure_survey.csv";
    std::ifstream s( input.c_str(), std::ifstream::in );
    if(!s.is_open()){
        std::cout << "While opening a file an error was encountered" << std::endl;
        exit(1);
    }
    else{
        std::cout << "File " << input << " is successfully opened" << std::endl;
    }

    std::string input2 = "Data_Supplementary_price.csv";
    std::ifstream s2( input2.c_str(), std::ifstream::in );
    if(!s2.is_open()){
        std::cout << "While opening a file an error was encountered" << std::endl;
        exit(1);
    }
    else{
        std::cout << "File " << input2 << " is successfully opened" << std::endl;
    }

    std::string line;
    int current_row = 0;
    int n_categories = 0;
    // Open first file
    while(std::getline( s, line )){
        std::stringstream ss( line );
        std::string category="";
        std::string sValue="";
        /// Break up line into components
        /// separated by ','
        int column_no = 0;
        // Initialize keys and values
        while(std::getline(ss, category, ',')){
            //std::cout << "Column: " << column_no << std::endl; 
            /// Category labels in the csv
            if(current_row==0) { 
                // Category as key, initialize NULL vector
                m_ConsExp.insert(std::make_pair<std::string, std::vector<float>* >(category.c_str(),0) ); 
                m_ConsExpIndex.insert(std::make_pair <int, std::string>((int&&)column_no,category.c_str()) ); 

                std::vector<float>* vptr = new std::vector<float>();
                m_ConsExp[category] = vptr;

                outputTree->Branch(std::string("ConsExp_"+category).c_str(), "std::vector<float>", &(m_ConsExp[category]) );

                ++n_categories;
                ++column_no;
            }
            else{

                sValue = category;
                if(strcmp(sValue.c_str(),"\\N")==0) sValue = "-9999";
                std::stringstream ssValue(sValue);
                float value;
                ssValue >> value;
                //std::cout << "Column NewID value: " << value << std::endl; 
                //m_ConsExp["newid"]->push_back(value);
                fillVector(column_no, value, m_ConsExp, m_ConsExpIndex);
                column_no++;

            }
        }

        current_row++;
    }
    
    std::cout << "Number of Categories in " << input << ": " << n_categories << std::endl;
    std::cout << "Number of Rows in " << input << ": " << current_row << std::endl;

    current_row = 0;
    n_categories = 0;
    /// Second file
    while(std::getline( s2, line )){
        std::stringstream ss( line );
        std::string category="";
        std::string sValue="";
        /// Break up line into components
        /// separated by ','
        int column_no = 0;
        // Initialize keys and values
        while(std::getline(ss, category, ',')){
            //std::cout << "Column: " << column_no << std::endl; 
            /// Category labels in the csv
            if(current_row==0) { 
                // Category as key, initialize NULL vector
                m_Prices.insert(std::make_pair<std::string, std::vector<float>* >(category.c_str(),0) ); 
                m_PricesIndex.insert(std::make_pair <int, std::string>((int&&)column_no,category.c_str()) ); 

                std::vector<float>* vptr = new std::vector<float>();
                m_Prices[category] = vptr;

                outputTree->Branch(std::string("Prices_"+category).c_str(), "std::vector<float>", &(m_Prices[category]) );

                ++n_categories;
                ++column_no;
            }
            else{

                sValue = category;
                if(strcmp(sValue.c_str(),"\\N")==0) sValue = "-9999";
                std::stringstream ssValue(sValue);
                float value;
                ssValue >> value;
                //std::cout << "Column NewID value: " << value << std::endl; 
                //m_Prices["newid"]->push_back(value);
                fillVector(column_no, value, m_Prices, m_PricesIndex);
                column_no++;

            }
        }

        current_row++;
    }
    outputTree->Fill();
    
    std::cout << "Number of Categories in " << input2 << ": " << n_categories << std::endl;
    std::cout << "Number of Rows in " << input2 << ": " << current_row << std::endl;

    // Fill histos
    TH1F* h_age = new TH1F("h_age","h_age", 300, 0.0, 110.0);
    h_age->Sumw2();
    TH1F* h_edu = new TH1F("h_edu","h_edu", 300, 0.0, 60.0);
    h_edu->Sumw2();
    TH1F* h_race = new TH1F("h_race","h_race", 60, 0.0, 6.0);
    h_race->Sumw2();
    TH1F* h_sex = new TH1F("h_sex","h_sex", 20, 0.0, 3.0);
    h_sex->Sumw2();
    TH1F* h_region = new TH1F("h_region","h_region", 60, 0.0, 6.0);
    h_region->Sumw2();
    for(unsigned int iter=0; iter<m_ConsExp["newid"]->size(); iter++){
    /*    std::cout << "newid: " << m_ConsExp["newid"]->at(iter) << std::endl;
        std::cout << "age: " << m_ConsExp["age"]->at(iter) << std::endl;
        std::cout << "incoll: " << m_ConsExp["incoll"]->at(iter) << std::endl;
      */  
        h_age->Fill(m_ConsExp["age"]->at(iter));
        h_edu->Fill(m_ConsExp["educatio"]->at(iter));
        h_race->Fill(m_ConsExp["race"]->at(iter));
        h_sex->Fill(m_ConsExp["sex"]->at(iter));
        h_region->Fill(m_ConsExp["region"]->at(iter));


    }//iter
    for(unsigned int iter=0; iter<m_Prices["year"]->size(); iter++){
        //std::cout << "year: " << m_Prices["year"]->at(iter) << std::endl;
        
    }//iter
    

    // Write histos
    _fOut->Write();
    return 0;
}
