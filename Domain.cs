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
            //We do cTrials trials, in each trial we calculate the total reward, finally
            // the function returns the average reward
            for (int i=0; i<cTrials; i++)
            {
                //We sample an initial state
                State initialState = InitialBelief.sampleState();
                double curTrialReward = calcTrialReward(p, cStepsPerTrial, initialState, InitialBelief);
                accumulated_reward += curTrialReward;
            }
            return accumulated_reward / cTrials;
        }


         /**
         * a recursive function performing one trial with stepsLeft steps, we are given a policy p,
         * stepsLeft, current state state, and current belief state bs
         * 
         * a) compute the action for the belief state bs
         * b) sample the result of applying a to s, obtaining nextState.
         * c) sample an observation o based on a and nextState
         * d) compute the new belief state given your old belief state, a, and o.
         * e) call the function recursively with the same policy p, stepsLeft-1, the new state, and the new
         * belief state. finally we accumalate the reward.
         * 
         * 
         */
        private double calcTrialReward(Policy p, int stepsLeft, State state, BeliefState bs)
        {
            // If we are already in a goal state or no steps are left then the reward is 0
            if (IsGoalState(state) || stepsLeft == 0)
                return 0;

            //Calculating the action for the belief state based on the policy
            Action a = p.GetAction(bs);
            
            //applying a on state, resulting in a new state nextState
            State nextState = state.Apply(a);
            //The reward of performing a on state
            double reward = nextState.Reward(a);
            // We sample an observation based on nextState and a
            Observation o = nextState.RandomObservation(a);

            // Updating the reward, calling the function recursively so we continue the "forward search" to goal state 
            reward += this.DiscountFactor * calcTrialReward(p, stepsLeft-1, nextState, bs.Next(a,o));
            return reward;
        }

    }
}