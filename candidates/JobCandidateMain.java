package com.bham.pij.assignments.candidates;

public class JobCandidateMain {

    public static void main (String args[]){
        CleaningUp t1 = new CleaningUp();
        t1.cleanUpFile();
        CandidatesToInterview t2 = new CandidatesToInterview();
        t2.findCandidates();
        t2.candidatesWithExperience();
        t2.createCSVFile();
        t2.createReport();

}}
