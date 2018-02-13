using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    abstract class Domain
    {
        public abstract IEnumerable<State> States { get; }
        public abstract IEnumerable<Action> Actions { get; }
        public abstract IEnumerable<Observation> Observations { get; }
        public abstract BeliefState InitialBelief { get; }
        public abstract double MaxReward { get; }
        public abstract bool IsGoalState(State s);
        public abstract State GetState(int iStateIdx);
        public double DiscountFactor { get; protected set; }
        public double ComputeAverageDiscountedReward(Policy p, int cTrials, int cStepsPerTrial) // Consider recurisve implementation
        {
            double accumulated_reward = 0;
            for(int i=0; i<cTrials; i++)
            {
                State initialState = InitialBelief.sampleState();
                double curTrialReward = calcTrialReward(p, cStepsPerTrial, initialState, InitialBelief);
                accumulated_reward += curTrialReward;
            }
            return accumulated_reward / cTrials;
        }

        private double calcTrialReward(Policy p, int stepsLeft, State state, BeliefState bs)
        {
            if (IsGoalState(state) || stepsLeft == 0)
                return 0;

            Action a = p.GetAction(bs);
            double reward = state.Reward(a);
            State nextState = state.Apply(a);

            Observation o = nextState.RandomObservation(a);

            reward += this.DiscountFactor * calcTrialReward(p, stepsLeft--, nextState, bs.Next(a,o));
            return reward;
        }

        

    }
}

/**
 * this simulates your policy for a number of iterations multiple times, and computes the average reward obtained.
 * To generate a single trial you:
 * 1. Sample a starting state s from the initial belief state.
 * 2. Repeat until goal is reached
 *  a) compute the action for the belief state a
 *  b) sample the result of applying a to s, obtaining s'.
 *  c) sample an observation o based on a and s'
 *  d) compute the new belief state given your old belief state, a, and o.
 *  e) accumulate the reward
**/
