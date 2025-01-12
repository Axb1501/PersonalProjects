package com.bham.pij.assignments.pontoon;

import java.util.ArrayList;

public class Card {

    private Suit suit;
    private Value value;

    public Card(Suit suit, Value value){
        this.suit = suit;
        this.value = value;

    }
    public static enum Value {ACE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING;}




        public Value getValue(){
            return this.value;

        }

        public void setValue (Value value){
            this.value = value;

        }


    public static enum Suit {SPADES, CLUBS, HEARTS, DIAMONDS;}




        public Suit getSuit() {
            return this.suit;

        }

        public void setValue(Suit suit) {
            this.suit = suit;

        }


    public ArrayList<Integer> getNumericalValue(){
        ArrayList<Integer> numericalValue = new ArrayList<Integer>();
        switch (value) {
            case ACE:
                numericalValue.add(1);
                numericalValue.add(11);
                break;
            case TWO:
                numericalValue.add(2);
                break;
            case THREE:
                numericalValue.add(3);
                break;
            case FOUR:
                numericalValue.add(4);
                break;
            case FIVE:
                numericalValue.add(5);
                break;
            case SIX:
                numericalValue.add(6);
                break;
            case SEVEN:
                numericalValue.add(7);
                break;
            case EIGHT:
                numericalValue.add(8);
                break;
            case NINE:
                numericalValue.add(9);
                break;
            case TEN:
            case KING:
            case QUEEN:
            case JACK:
                numericalValue.add(10);

        }
        return numericalValue;

    }




}
