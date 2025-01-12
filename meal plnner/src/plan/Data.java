package plan;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Data {

    public int lineCounter() throws IOException {

        BufferedReader counting = new BufferedReader(new FileReader("C:\\Users\\aaron\\IdeaProjects\\meal plnner\\src\\plan\\recipes.txt"));
        String input;
        int count = 0;
        while ((input = counting.readLine()) != null) {
            count++;
        }
        return count;


    }

    public ArrayList<Integer> Randomizer(int output) {
        Random random = new Random();
        int meal1 = random.nextInt(output);
        int meal2 = random.nextInt(output);
        int meal3 = random.nextInt(output);
        int meal4 = random.nextInt(output);
        int meal5 = random.nextInt(output);
        int meal6 = random.nextInt(output);
        meal1 = meal1 + 1;
        meal2 = meal2 + 1;
        meal3 = meal3 + 1;
        meal4 = meal4 + 1;
        meal5 = meal5 + 1;
        meal6 = meal6 + 1;
        if (meal1 == meal2) {
            while (meal1 == meal2) {
                meal2 = random.nextInt(output);
            }
        }
        if (meal1 == meal3) {
            while (meal1 == meal3) {
                meal3 = random.nextInt(output);
            }
        }
        if (meal1 == meal4) {
            while (meal1 == meal4) {
                meal4 = random.nextInt(output);
            }
        }
        if (meal1 == meal5) {
            while (meal1 == meal5) {
                meal5 = random.nextInt(output);
            }
        }
        if (meal1 == meal6) {
            while (meal1 == meal6) {
                meal2 = random.nextInt(output);
            }
        }
        if (meal2 == meal3) {
            while (meal2 == meal3) {
                meal3 = random.nextInt(output);
            }
        }
        if (meal2 == meal4) {
            while (meal2 == meal4) {
                meal4 = random.nextInt(output);
            }
        }
        if (meal2 == meal5) {
            while (meal2 == meal5) {
                meal5 = random.nextInt(output);
            }
        }
        if (meal2 == meal6) {
            while (meal2 == meal6) {
                meal2 = random.nextInt(output);
            }
        }
        if (meal3 == meal4) {
            while (meal3 == meal4) {
                meal4 = random.nextInt(output);
            }
        }
        if (meal3 == meal5) {
            while (meal3 == meal5) {
                meal5 = random.nextInt(output);
            }
        }
        if (meal3 == meal6) {
            while (meal3 == meal6) {
                meal2 = random.nextInt(output);
            }
        }
        if (meal4 == meal5) {
            while (meal4 == meal5) {
                meal5 = random.nextInt(output);
            }
        }
        if (meal4 == meal6) {
            while (meal4 == meal6) {
                meal2 = random.nextInt(output);
            }
        }
        if (meal5 == meal6) {
            while (meal5 == meal6) {
                meal2 = random.nextInt(output);
            }
        }
        ArrayList<Integer> meals = new ArrayList<Integer>();
        meals.add(meal1);
        meals.add(meal2);
        meals.add(meal3);
        meals.add(meal4);
        meals.add(meal5);
        meals.add(meal6);
        return meals;
    }

    public ArrayList<Integer> Calculator(ArrayList<Integer> number) {

        int protein = 0;
        int carbs = 0;
        int fat = 0;
        int total = 0;
        int i = 0;
        while (number.size() > i) {
            int calorie = number.get(i);
            i = i + 1;
            try (Scanner reader = new Scanner(new FileReader("C:\\Users\\aaron\\IdeaProjects\\meal plnner\\src\\plan\\recipes.txt"))) {
                for (int j = 0; j < calorie - 1; j = j + 1) {
                    reader.nextLine();
                }
                String line = reader.nextLine();
                String[] parts = line.split(",");
                String count1 = parts[1];
                String count2 = parts[2];
                String count3 = parts[3];
                String count4 = parts[4];
                carbs = carbs + Integer.parseInt(count1);
                fat = fat + Integer.parseInt(count2);
                protein = protein + Integer.parseInt(count3);
                total = total + Integer.parseInt(count4);


            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        ArrayList<Integer> calories = new ArrayList<>();
        calories.add(carbs);
        calories.add(fat);
        calories.add(protein);
        calories.add(total);
        return calories;


    }

    public String line(int lineNumber) {
        String line = null;
        try (Scanner reader = new Scanner(new FileReader("C:\\Users\\aaron\\IdeaProjects\\meal plnner\\src\\plan\\recipes.txt"))) {
            for (int j = 0; j < lineNumber - 1; j = j + 1) {
                reader.nextLine();
            }
            line = reader.nextLine();

        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
        return line;

    }
}
