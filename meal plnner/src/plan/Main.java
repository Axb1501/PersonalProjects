package plan;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class Main {

    public static void main(String args[]) throws IOException {
        Data t1 = new Data();
        int output = t1.lineCounter();
        ArrayList<Integer> number = t1.Randomizer(output);
        ArrayList<Integer> rerun = new ArrayList<>();
        int i = 0;
        while (number.size() > i) {
            int answer = number.get(i);
            i = i + 1;
            System.out.println(answer);
        }
        ArrayList<Integer> calories = t1.Calculator(number);
        if (calories.get(0) < 800 || calories.get(0) > 900 || calories.get(1) < 800 || calories.get(1) > 900 || calories.get(2) < 800 || calories.get(2) > 900  ) {
            while (calories.get(0) < 800 || calories.get(0) > 900 || calories.get(1) < 800 || calories.get(1) > 900 || calories.get(2) < 800 || calories.get(2) > 900 ) {
               System.out.println(calories);
                rerun = t1.Randomizer(output);
                calories = t1.Calculator(rerun);
            }
        }
        for (int j = 0; number.size() > j ; j = j + 1){
            String meal = t1.line(number.get(j));
            String [] parts = meal.split(",");
            String count1 = parts[0];
            String count2 = parts[1];
            String count3 = parts[2];
            String count4 = parts[3];
            String count5 = parts[4];
            System.out.println("meal: " + j + 1);
            System.out.println("Recipe: " + count1);
            System.out.println("Carbohydrates: " + count2);
            System.out.println("Fat: " + count3);
            System.out.println("Protein: " + count4);
            System.out.println("Calories: " + count5);
            System.out.println("");
        }
        System.out.println("total");
        System.out.println("Carbohydrates: " + calories.get(0));
        System.out.println("Fat: " + calories.get(1));
        System.out.println("Protein: " + calories.get(2));
        System.out.println("Calories: " + calories.get(3));




    }




}
