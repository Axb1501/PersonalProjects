package com.bham.pij.assignments.candidates;
import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class CleaningUp {







    public void cleanUpFile(){




        try (Scanner reader = new Scanner(new FileReader("dirtycv.txt"))) {
            String number = "";
            String cleanse = "";
            ArrayList<String> appender = new ArrayList<>();
            while (reader.hasNextLine()){
                String line = reader.nextLine();

                if (line.contains("End")){

                    for (String s: appender){
                        cleanse = cleanse + s + ",";
                    }

                    cleanse = cleanse + "\n";
                    appender.clear();


                }
                else if (!line.equals("End")){
                    String[] parts = line.split(":");
                    if (parts[0].equals("Qualification") || parts[0].equals("Position") || parts[0].equals("Experience")|| parts[0].equals("eMail")){
                        appender.add(parts[1]);}
                    if (parts[0].contains("CV")){
                        number = parts[0].replace("CV ", "");
                    }
                    else if (parts[0].equals("Surname")){
                        for (int i = number.length(); i < 4; i++){
                            String zero = "0";
                            number = zero.concat(number) ;
                        }
                        appender.add(parts[1].concat(number));
                    }
                }
            } reader.close();


            try { FileWriter writer = new FileWriter("cleancv.txt");
                writer.write(cleanse.substring(0,cleanse.length() - 1));
                writer.close();}

            catch
            (IOException e)
            { System.out.println("error");}





        } catch (FileNotFoundException e) {
            System.out.println("error");
            e.printStackTrace();
        }


    }

}
