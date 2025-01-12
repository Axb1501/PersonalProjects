package com.bham.pij.assignments.pontoon;

import java.util.*;
import java.util.Random;

import java.util.ArrayList;

public class Deck {
    ArrayList<Card> deck = new ArrayList<Card>();

    public Deck(){
        reset();
    }

    public void reset() {
        deck.clear();
        int i = 0;
        for (Card.Suit s : Card.Suit.values()) {
            for (Card.Value v : Card.Value.values()) {
                deck.add(new Card(s, v));
                i++;
            }
        }
        shuffle();
    }

    public void shuffle(){
        Collections.shuffle(deck);
    }

    public Card getCard(int i){
        return deck.get(i);
    }

    public Card dealRandomCard(){
        Random random = new Random();
        int upperbound = deck.size();
        int int_random = random.nextInt(upperbound);
        Card randomCard = deck.get(int_random);
        deck.remove(int_random);
        return randomCard;

    }

    public int size(){
        int deck_size = deck.size();
        return deck_size;
    }







}
