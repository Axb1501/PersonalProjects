package com.bham.pij.assignments.pontoon;

import java.util.ArrayList;


public class Player {
    ArrayList<Card> Players_hand = new ArrayList<Card>();

    private String Player_name;

    public Player (String name){
        String Player_name = name;
    }

    public void dealToPlayer( Card card){
        Players_hand.add(card);
    }

    public String getName(){
        return Player_name;
    }

    public void removeCard(Card card){
        Players_hand.remove(card);
    }

    public ArrayList<Integer> getNumericalHandValue() {
        ArrayList<Integer> handValue = new ArrayList<Integer>();
        ArrayList<Integer> numerical = new ArrayList<Integer>();
        int numericalSum = 0;
        int j = 0;
        for (int i = 0; i < Players_hand.size() ; i++) {
            Card test = Players_hand.get(i);
            numerical = test.getNumericalValue();
            if (numerical.size() == 2) {
                j = j + 1;
            } else {
                numericalSum = numerical.get(0) + numericalSum;

            }

        }
        if (j == 1) {
            handValue.add(numericalSum + 1);
            handValue.add(numericalSum + 11);

        }
        else if (j == 2) {
            handValue.add(numericalSum +2);
            handValue.add(numericalSum + 12);
            handValue.add(numericalSum + 22);

        }
        else if (j == 3){
            handValue.add(numericalSum + 3);
            handValue.add(numericalSum + 13);
            handValue.add(numericalSum + 23);
            handValue.add(numericalSum + 33);

        }
        else if (j == 4){
            handValue.add(numericalSum + 4);
            handValue.add(numericalSum + 14);
            handValue.add(numericalSum + 24);
            handValue.add(numericalSum + 34);
            handValue.add(numericalSum + 44);


        }
        else{
            handValue.add(numericalSum);

        }
        return handValue;
    }

    public int getBestNumericalHandValue(){
        int highest_value = 0;
        ArrayList<Integer> handValue = getNumericalHandValue();
        for (int i = 0; i < handValue.size(); i++){
            if(handValue.get(i) > highest_value && handValue.get(i) <= 21){
                highest_value = handValue.get(i);
            }
        }
        return highest_value;


    }

    public ArrayList<Card> getCards(){
        return Players_hand;

    }

    public int getHandSize(){
        int size = Players_hand.size();
        return size;
    }


}
