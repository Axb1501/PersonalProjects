
package com.bham.pij.assignments.pontoon;

public class Pontoon extends CardGame {

    public Pontoon(int nplayers) {
        super(nplayers);
    }

    public void dealInitialCards() {
       for (Player player: Players ){
           player.dealToPlayer(cards.dealRandomCard());
           player.dealToPlayer(cards.dealRandomCard());

       }


    }

    public int compareHands(Player hand1, Player hand2) {
        if (isPontoon(hand1) == true) {
            if (isPontoon(hand2) == false) {
                return -1;
            } else if (isPontoon(hand2) == true) {
                return 0;
            }
        }
        if (isPontoon(hand2) == true) {
            if (isPontoon(hand1) == false) {
                return 1;
            } else if (isPontoon(hand1) == true)
                return 0;
        }
        if (fiveCards(hand1) == true) {
            if (fiveCards(hand2) == true) {
                return 0;
            } else if (fiveCards(hand2) == false) {
                return -1;
            }
        }
        if (fiveCards(hand2) == true) {
            if (fiveCards(hand1) == true) {
                return 0;
            } else if (fiveCards(hand1) == false) {
                return 1;
            }
        }
        if (twentyOne(hand1) == true) {
            if (twentyOne(hand2) == true) {
                return 0;
            }
            if (twentyOne(hand2) == false) {
                return -1;
            }
        }
        if (twentyOne(hand2) == true) {
            if (twentyOne(hand1) == true) {
                return 0;
            }
            if (twentyOne(hand1) == false) {
                return 1;
            }
        }
        if (hand1.getBestNumericalHandValue() > hand2.getBestNumericalHandValue()) {
            return -1;
        }
        if (hand1.getBestNumericalHandValue() == hand2.getBestNumericalHandValue()) {
            return 0;
        }
        else {
            return 1;
        }
    }



    public boolean isPontoon(Player p) {
        if (p.getHandSize() == 2 && p.getBestNumericalHandValue() == 21){
            return true;
        }
        else{
            return false;
        }

    }

    public boolean fiveCards(Player p){
    if (p.getHandSize() == 5 && p.getBestNumericalHandValue() <= 21){
        return true;
    }
    else{
        return false;
    }
}
    public boolean twentyOne(Player p){
        if(p.getBestNumericalHandValue() == 21){
            return true;
        }
        else{
            return false;
        }
    }





}




