using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace POMDP
{
    class PointBasedValueIteration : Policy
    {
        private Domain m_dDomain;
        private List<AlphaVector> m_lVectors;
        //private Dictionary<KeyValuePair<BeliefState, KeyValuePair<Action, Observation>>, BeliefState> m_dNextBeliefState;
        private Dictionary<Tuple<AlphaVector, Action, Observation>, AlphaVector> m_dAlphaA_O;
        private Dictionary<Action, AlphaVector> rewardsVectors;

        public PointBasedValueIteration(Domain d)
        {
            m_dDomain = d;
            m_lVectors = new List<AlphaVector>();
            m_dAlphaA_O = new Dictionary<Tuple<AlphaVector, Action, Observation>, AlphaVector>();
            m_lVectors.Add(createInitialAlphaVector()); //Adding an initial alpha vector
            rewardsVectors = new Dictionary<Action, AlphaVector>();
        }

        /**
        * Calculated the best action for bs w.r.t to m_lVectors
        * Basically translates m_lVectors into a policy 
        */
        public override Action GetAction(BeliefState bs)
        {
            AlphaVector avBest = null;
            ValueOf(bs, m_lVectors, out avBest);
            return avBest.Action;
        }

        /**
         *  The Backup operation, receives a belief state bs, and returns 
         *  Backup(m_lVectors,bs)
         * (The best alpha vector alpha_a_bs maximizing
         * dot(b,alpha_a_bs))
         * 
         */
        private AlphaVector backup(BeliefState bs)
        {
            AlphaVector avBest = new AlphaVector(), avCurrent = new AlphaVector();
            double dMaxValue = double.NegativeInfinity, dValue = 0.0;

            //We loop over all actions in domain, and for every action a 
            //we take the best alpha vector with a on its root
            foreach (Action a in m_dDomain.Actions)
            {
                avCurrent = computeBestAlpha(a, bs); // alpha_a_b
                dValue = avCurrent.InnerProduct(bs); // dot product with bs
                if (dValue > dMaxValue)
                {// taking the vector alpha_a_b that maximizes the dot product
                    avBest = avCurrent;
                    dMaxValue = dValue;
                }
            }
            return avBest; //returns the best alpha_a_bs
        }

        /**
         * Computes the best alpha vector with action on root for belief state bs
         * (alpha_action_bs)
         */
        private AlphaVector computeBestAlpha(Action action, BeliefState bs)
        {
            //initializing an alpha vector with action on its root
            AlphaVector discountedRewardVector = new AlphaVector(action);

            // We loop over all observations and alpha vectors for each observation obs, 
            // we find the alpha vector maximizing dot(bs,alpha_action_obs) - we will use 
            // these vectors (their sum) in order to calculate alpha_a_b
            foreach (Observation obs in m_dDomain.Observations)
            { //We compute alpha_a_o for every observation o, according to the equation in the slides
                AlphaVector cur_alpha_ao = null;
                AlphaVector best_alpha_ao = new AlphaVector();
                double best_val = double.NegativeInfinity;  double cur_val = 0;

                //Looping over all alpha vectors, finding the best alpha that maximizes dot(bs,alpha_action_obs)
                foreach (AlphaVector av in m_lVectors)
                {
                    //We compute av_action_obs for every av
                    cur_alpha_ao = computeAlphaAO(av, action, obs);
                    // dot product between av_action_obs abd the belief state bs
                    cur_val = cur_alpha_ao.InnerProduct(bs);
                    //We take the vector maximizing the dot product
                    if (cur_val > best_val) {
                        best_alpha_ao = cur_alpha_ao;
                        best_val = cur_val;
                    }  
                }
                //We compute the sum of these vectors, (SUM(arg max(dot(bs,alpha_bs_a))))
                discountedRewardVector += best_alpha_ao;
            }
            // Multiplying it with the discount factor
            discountedRewardVector = discountedRewardVector * m_dDomain.DiscountFactor;

            AlphaVector rA; //action's rewards vector, We add it to the sum, and return the result
            if (rewardsVectors.ContainsKey(action))
            {
                rA = rewardsVectors[action];
            }
            else
            {
                rA = new AlphaVector();
                foreach (State s in m_dDomain.States)
                    rA[s] = s.Reward(action);
                rewardsVectors[action] = rA;
            }
            
            return discountedRewardVector + rA;
        }

        /**
         * Receives alpha vector alpha, action a, and observation o.
         * calculates and returns alpha_a_o 
         * (alpha_a_o[s]=SUM(alpha[s']*O(a,s',o)*T(s,a,s')))
         */
        private AlphaVector computeAlphaAO(AlphaVector alpha, Action a, Observation o)
        {
            Tuple<AlphaVector, Action, Observation> key = new Tuple<AlphaVector, Action, Observation>(alpha, a, o);
            if (m_dAlphaA_O.ContainsKey(key))
                return m_dAlphaA_O[key];

            else
            {
                AlphaVector res = new AlphaVector(a);
                //We loop over all states s, for each s we compute alpha_a_o[s]
                foreach (State s in m_dDomain.States)
                {
                    double accumulated_sum = 0;
                    res[s] = 0;
                    //Looping only on successors of s because only for them T(s,a,succ)>0
                    foreach (State succ in s.Successors(a))
                        accumulated_sum += (alpha[succ] * succ.ObservationProbability(a, o) * s.TransitionProbability(a, succ));
                    res[s] = accumulated_sum;
                }
                m_dAlphaA_O.Add(key, res);
                return res;
            }
        }

        private List<BeliefState> SimulateTrial(Policy p, int cMaxSteps)
        {
            BeliefState bsCurrent = m_dDomain.InitialBelief, bsNext = null;
            State sCurrent = bsCurrent.sampleState(), sNext = null;
            Action a = null;
            Observation o = null;
            List<BeliefState> lBeliefs = new List<BeliefState>();
            while (!m_dDomain.IsGoalState(sCurrent) && lBeliefs.Count < cMaxSteps)
            {
                a = p.GetAction(bsCurrent);
                sNext = sCurrent.Apply(a);
                o = sNext.RandomObservation(a);
                bsNext = bsCurrent.Next(a, o);
                bsCurrent = bsNext;
                lBeliefs.Add(bsCurrent);
                sCurrent = sNext;
            }
            return lBeliefs;
        }

        private List<BeliefState> CollectBeliefs(int cBeliefs)
        {
            Debug.WriteLine("Started collecting " + cBeliefs + " points");
            RandomPolicy p = new RandomPolicy(m_dDomain);
            int cTrials = 100, cBeliefsPerTrial = cBeliefs / cTrials;
            List<BeliefState> lBeliefs = new List<BeliefState>();
            while (lBeliefs.Count < cBeliefs)
            {
                lBeliefs.AddRange(SimulateTrial(p, cBeliefsPerTrial));
            }
            Debug.WriteLine("Collected " + lBeliefs.Count + " points");
            return lBeliefs;
        }

        /**
         * Calculates the value of a belief state bs w.r.t a list to alpha vectors.
         * i.e finds the alpha vector alpha that maximizes dot(bs,alpha), returns the value of 
         * this dot product, and return the vector as avBest
         * 
         * 
         */ 
        private double ValueOf(BeliefState bs, List<AlphaVector> lVectors, out AlphaVector avBest)
        {
            double dValue = 0.0, dMaxValue = double.NegativeInfinity;
            avBest = null;
            //We loop over all alpha vectors
            foreach (AlphaVector av in lVectors)
            {
                dValue = av.InnerProduct(bs);
                if (dValue > dMaxValue) //taking the maximum dot product
                {
                    dMaxValue = dValue;
                    avBest = av;
                }
            }
            return dMaxValue;
        }

        /**
         * Initializes an alpha vector with some best practice presented in the provided article 
         */
        private AlphaVector createInitialAlphaVector()
        {
            AlphaVector V0 = new AlphaVector();
            double minReward = Double.PositiveInfinity;
            foreach (State s in m_dDomain.States)
            {
                foreach (Action a in m_dDomain.Actions)
                {
                    if (minReward > s.Reward(a))
                        minReward = s.Reward(a);
                }
            }
            double defaultVal = (1 / (1 - m_dDomain.DiscountFactor)) * minReward; //best practice
            foreach (State s in m_dDomain.States)
            {
                V0[s] = defaultVal;
            }

            return V0;
        }

        /**
         * Performs the Value Iteration algorithm using the Perseus update algorithm,
         * generates a set containing cBelief belief states, and performs value iterations for 
         * maximum cMaxIterations
         *
         */
        public void PointBasedVI(int cBeliefs, int cMaxIterations)
        {
            //Generates an initial set containing cBelief belief states
            List<BeliefState> beliefStates = CollectBeliefs(cBeliefs);
            List<AlphaVector> vTag; //V' 

            //const double EPSILON = 0.1; //The convergence boundry
            int iterationsLeft = cMaxIterations;

            while (iterationsLeft > 0)
            {
                vTag = new List<AlphaVector>();
                List<BeliefState> beliefStatesLeftToImprove= new List<BeliefState>(beliefStates); // B'
                while (beliefStatesLeftToImprove.Count() > 0)
                { //While there are belief states to improve
                    //Console.WriteLine("Improvable belief states left");
                    //Console.WriteLine(beliefStatesLeftToImprove.Count());

                    //selecting a random index of a belief state to improve
                    int ri = RandomGenerator.Next(beliefStatesLeftToImprove.Count());

                    //We want to iterate over the belief states set and recieve the ri'th item
                    List<BeliefState>.Enumerator e = beliefStatesLeftToImprove.GetEnumerator();
                    for(int i=0; i<ri+1; i++) //iterating until the belief state at index ri
                        e.MoveNext();
                    BeliefState sampledBS = e.Current;//samplesBS is a randomly chosen belief state to for improvement

                    //Console.WriteLine("Iterations left: " + iterationsLeft);
                    //Console.WriteLine("Improvable bs left: " + beliefStatesLeftToImprove.Count());

                    //We calculate the backup of samplesBS
                    AlphaVector alpha = backup(sampledBS);
                    AlphaVector alphaToAdd;//It will contain the alpha vector to add to V'

                    AlphaVector prevBestAlphaVector = null;
                    //calculating the value of sampledBS (V(samplesBS)) which is the best dot product alpha*b
                    double prevValue = ValueOf(sampledBS, m_lVectors,out prevBestAlphaVector);
                    
                    if (alpha.InnerProduct(sampledBS) >= prevValue) // alpha is dominating, remove all belief states that are improved by it
                    {
                        //Console.WriteLine("Found an improving vec");
                        List<BeliefState> beliefStatesToKeep = new List<BeliefState>();
                        foreach (BeliefState b_prime in beliefStatesLeftToImprove)
                        {
                            AlphaVector a = null;
                            if (alpha.InnerProduct(b_prime) < ValueOf(b_prime, m_lVectors, out a)) 
                                beliefStatesToKeep.Add(b_prime);
                        }
                        beliefStatesLeftToImprove = beliefStatesToKeep;
                        //In the case alpha is dominating, we add alpha to V' 
                        alphaToAdd = alpha;
                    }

                    else
                    { //alpha does not improve,we remove sampledBS from the set
                        beliefStatesLeftToImprove.Remove(sampledBS);
                        alphaToAdd = prevBestAlphaVector;
                    }

                    if (!vTag.Contains(alphaToAdd))
                        vTag.Add(alphaToAdd);
                   
                }

                /**
                //We estimate how the alpha vectors set was changed
                double diff = estimateDiff(m_lVectors, vTag, beliefStates);
                Console.WriteLine(diff);

                //The difference between the current set, and the previous is less than epsilon
                //We finish the update algorithm
                //if (diff < EPSILON)
                //   break;
                **/

                Console.WriteLine("Iterations left {0}", iterationsLeft);
                m_lVectors = vTag;
                iterationsLeft--;
            }
        }

        /**
        // Calculate the convergence criteria between two alpha vectors sets w.r.t the beliefStates set, as stated in the article
        private double estimateDiff(List<AlphaVector> s1, List<AlphaVector> s2, List<BeliefState> beliefStates)
        {
            double maxDiff = Double.NegativeInfinity;
            foreach (BeliefState b in beliefStates)
            {
                AlphaVector a1, a2 = null;
                double curDiff = Math.Abs(ValueOf(b, s1, out a1) - ValueOf(b, s2, out a2));
                if (curDiff > maxDiff)
                    maxDiff = curDiff;
            }
            return maxDiff;
        }
        **/

    }
}
