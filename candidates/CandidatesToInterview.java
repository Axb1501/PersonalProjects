package com.bham.pij.assignments.candidates;
import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class CandidatesToInterview {
    String [] keywordsDegree = {"Degree in Computer Science", "Masters in Computer Science"};
    String [] keywordsExperience = {"Data analyst", "Programmer", "Computer programmer", "Operator" };



    public void findCandidates(){

        String all ="";


        try (Scanner reader = new Scanner(new FileReader("cleancv.txt"))){
            while (reader.hasNextLine()){
                String line = reader.nextLine();
                if ((line.contains(keywordsDegree[0]) || line.contains(keywordsDegree[1])) && (line.contains(keywordsExperience[0])
                || line.contains(keywordsExperience[1]) || line.contains(keywordsExperience[2]) || line.contains(keywordsExperience[3]))){
                    all = all + line.replace(",", " ");
                    all = all + "\n";
                    }
                } reader.close();

                }

        catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        try { FileWriter interviews = new FileWriter("to-Interview.txt");
            interviews.write(all.substring(0,all.length() - 1 ));
            interviews.close();}

        catch (FileNotFoundException e) {
            System.out.println("error");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void candidatesWithExperience(){
        String getLine = "";
        try (Scanner reader = new Scanner(new FileReader("to-interview.txt"))){
            while (reader.hasNextLine()){
                String line = reader.nextLine();
                String [] parts = line.split(" ");
                for (int i = 0; i < 8; i++ ){
                    try {
                        if (Double.parseDouble(parts[i]) > 5){
                                getLine = getLine + parts[0] + " " + parts[i] + "\n";



                        }


                    } catch(NumberFormatException e){
                    }

                }


            } reader.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        try { FileWriter interviewExp = new FileWriter("to-interview-experience.txt");
            interviewExp.write(getLine.substring(0,getLine.length() - 1 ));
            interviewExp.close();}

        catch (IOException e) {
            e.printStackTrace();
        }


    }

    public void createCSVFile(){
        String previousRole = " ";
        String experience2 = " ";
        String currentRole = "";
        String experience = "";
        String all = "";
        try (Scanner reader = new Scanner(new FileReader("to-interview.txt"))){
            while (reader.hasNextLine()){
                int i = 0;
                String line = reader.nextLine();
                String [] parts = line.split(" ");
                String surname = parts[0] + ",";
                String qualification = parts[1] + " " + parts[2]+ " " + parts[3]+ " " + parts[4] + ",";
                String email = parts[parts.length - 1] +"\n";
                if (parts[5].equals("Data") || parts[5].equals("Computer")){
                    currentRole = parts[5] + " " + parts[6] + ",";
                    experience = parts [7] + ",";
                    i = 8;
                }
                else{
                    currentRole = parts[5] + ",";
                    experience = parts [6] + ",";
                    i = 7;
                }
                if (parts[i].equals(parts[parts.length - 1]) == false){
                    if (parts[i].equals("Data") || parts[i].equals("Computer")){
                        previousRole = parts[i] + " " + parts[i + 1] + ",";
                        experience2 = parts [i + 2] + ",";
                }
                    else{
                        previousRole = parts[i] + ",";
                        experience2 = parts [i + 1] + ",";
                    }

            }
                all = all + surname + qualification + currentRole + experience + previousRole + experience2 + email;
                previousRole = "" + ",";
                experience2 = "" + ",";
                currentRole = "" + ",";
                experience = "" + ",";




            } reader.close();
            try { FileWriter table = new FileWriter("to-interview-table-format.csv");
                table.write("Identifier"+","+"Qualification" + "," + "Position1" + "," + "Experience1" +"," + "Position2" + "," + "Experience2" +","+ "email" + "\n");
                table.write(all.substring(0,all.length() - 1 ));
                table.close();} catch (IOException e) {
                e.printStackTrace();
            }


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void createReport(){
        int i = 0;
        try (Scanner reader = new Scanner(new FileReader("to-interview-table-format.csv"))){
            while (reader.hasNextLine()) {
                String line = reader.nextLine();
                String[] parts = line.split(",");
                if (i == 0){
                    System.out.printf("%-20s%-40s%-25s%-20s%-38s", parts[0], parts[1], "Position", "Experience", "eMail");
                    i = i + 1;
                    continue;
                }
                System.out.printf("%n%-20s%-40s%-25s%-20s%-38s", parts[0], parts[1], parts[2], parts[3], parts[6]);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }




}
