package com.bham.pij.assignments.pontoon;

import java.util.ArrayList;

public abstract class CardGame {
    ArrayList<Player> Players = new ArrayList<>();
    public Deck cards;


    public CardGame(int nplayers) {
        for (int i = 1; i < nplayers + 1; i++){
            Players.add(new Player("Player_" + (i)));
        }
        cards = new Deck();

    }

    public abstract void dealInitialCards();

    public abstract int compareHands(Player hand1, Player hand2);

    public Deck getDeck() {
        return cards;
    }

    public Player getPlayer(int i){
        return Players.get(i);




    }

    public int getNumPlayers(){
        return Players.size();

    }

}




















