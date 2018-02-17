using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

namespace POMDP
{
    class BeliefState
    {
        private Dictionary<State,double> m_dBeliefs;
        private Domain m_dDomain;
        
        public double this[State s]
        {
            get 
            { 
                if(m_dBeliefs.ContainsKey(s))    
                    return m_dBeliefs[s];
                return 0.0;
            }
            set { m_dBeliefs[s] = value; }
        }

        public IEnumerable<KeyValuePair<State, double>> Beliefs(double dMin)
        {
            foreach(KeyValuePair<State, double> p in m_dBeliefs)
                if (p.Value >= dMin)
                    yield return p;
        }

        public BeliefState(Domain d)
        {
            m_dDomain = d;
            m_dBeliefs = new Dictionary<State, double>();
        }

        private void AddBelief(State s, double dProb)
        {
            if (!m_dBeliefs.ContainsKey(s))
                m_dBeliefs[s] = 0;
            m_dBeliefs[s] += dProb;
        }

        public BeliefState Next(Action a, Observation o)
        {
            BeliefState bsNext = new BeliefState(m_dDomain); //Represents the new belief state b_o_s

            double normalizing_factor = 0; //We will divide our resulted belief state by this factor, instead of calculating Pr(o|a,b)

            HashSet<State> reachableStates = new HashSet<State>();

            // The neighboring states are the union of all neighboring states 
            // of states with positive probability on current belief state.
            // When we calculate the new distribution over states, we just need 
            // to look on S' such that Tr(S,a,S')>0
            foreach (KeyValuePair<State, double> entry in m_dBeliefs)
            {
                if (entry.Value > 0)
                {
                    foreach (State s in entry.Key.Successors(a))
                    {
                        reachableStates.Add(s);
                        // We optimize the calculation by adding the weighted transition value as we build the reachableStates Set
                        // Instead of first calculating the set and only then finding all its ancenstors and perform the calculation
                        bsNext.AddBelief(s, entry.Value * entry.Key.TransitionProbability(a, s));
                    }

                }
            }

            foreach(State s_prime in reachableStates)
            {
                double trans_prob = 0;
                double obs_prob = s_prime.ObservationProbability(a, o); // We Calculate O(o,s',a)*(b\dot\Tr(s',a))
                trans_prob = bsNext[s_prime];
                // for each state s_prime trans_prob equals O(s_prime,a,o)*dot(b,Tr(s,a,s_prime))
                trans_prob *= obs_prob;
                //The normalizing factor is sum of all values, we divide the vector by this number to make it a distribution
                normalizing_factor += trans_prob;
                // Updating the new belief state
                bsNext[s_prime] = trans_prob;
            }

            foreach(State s in reachableStates)
                bsNext[s] /= normalizing_factor;

            Debug.Assert(bsNext.Validate());
            return bsNext;
        }

        public override string ToString()
        {
            string s = "<";
            foreach (KeyValuePair<State, double> p in m_dBeliefs)
            {
                if( p.Value > 0.01 )
                    s += p.Key + "=" + p.Value.ToString("F") + ",";
            }
            s += ">";
            return s;
        }

        public bool Validate()
        {
            //validate that every state appears at most once
            List<State> lStates = new List<State>(m_dBeliefs.Keys);
            int i = 0, j = 0;
            for (i = 0; i < lStates.Count; i++)
            {
                for (j = i + 1; j < lStates.Count; j++)
                {
                    if (lStates[i].Equals(lStates[j]))
                        return false;
                }
            }
            double dSum = 0.0;
            foreach (double d in m_dBeliefs.Values)
                dSum += d;
            if (Math.Abs(1.0 - dSum) > 0.001)
                return false;
            return true;
        }

        public double Reward(Action a)
        {
            double dSum = 0.0;
            foreach (KeyValuePair<State, double> p in m_dBeliefs)
            {
                dSum += p.Value * p.Key.Reward(a);
            }
            return dSum;
        }

        /**
        * Samples a State from the belief state distribution and returns it
        */
        public State sampleState()
        {
            // We sample a random number in [0,1]
            double dRnd = RandomGenerator.NextDouble();
            double dProb = 0.0;
            // We keep summing the probabilities until we reach the first state where the sum >= the random number
            // It is like having a set of disjoint intervals in [0,1] and we random a number, and see its interval 
            // As the length of the interval is larger (large prob), so the probability we picks a number in this interval.
            foreach (KeyValuePair<State, double> sd in Beliefs(0))
            {
                dProb = sd.Value;
                dRnd -= dProb;
                if (dRnd <= 0) // the sum reached dRnd
                    return sd.Key;
            }
            return Beliefs(0).Last().Key; // shouldn't get here...
        }

    }
}
