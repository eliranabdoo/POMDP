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
        private Dictionary<AlphaVector, Dictionary<Action, Dictionary<Observation, AlphaVector>>> m_dGCache;
        private Dictionary<KeyValuePair<BeliefState, KeyValuePair<Action, Observation>>, BeliefState> m_dNextBeliefState;

        public PointBasedValueIteration(Domain d)
        {
            m_dDomain = d;
            m_lVectors = new List<AlphaVector>();
            m_lVectors.Add(new AlphaVector()); //Adding the null plan alpha vector
            /**AlphaVector av = new AlphaVector();
            IEnumerator<Action> eA = m_dDomain.Actions.GetEnumerator();
            eA.MoveNext();
            Action a = eA.Current;
            
            foreach (State s in m_dDomain.States)
                av[s] = s.Reward(a);
            m_lVectors.Add(av);**/
        }

        public override Action GetAction(BeliefState bs)
        {
            AlphaVector avBest = null;
            ValueOf(bs, m_lVectors, out avBest);
            return avBest.Action;
        }


        private AlphaVector alpha_a_o(Action a, Observation o, AlphaVector av)
        {
            if (!m_dGCache.ContainsKey(av))
                m_dGCache[av] = new Dictionary<Action, Dictionary<Observation, AlphaVector>>();
            if (!m_dGCache[av].ContainsKey(a))
                m_dGCache[av][a] = new Dictionary<Observation, AlphaVector>();
            if (m_dGCache[av][a].ContainsKey(o))
                return m_dGCache[av][a][o];
            AlphaVector avNew = new AlphaVector(a);
            foreach (State s in m_dDomain.States)
            {
                double dSum = 0.0;
                foreach (State sTag in m_dDomain.States)
                {
                    dSum += sTag.ObservationProbability(a, o) * s.TransitionProbability(a, sTag) * av[sTag];

                }
                avNew[s] = dSum;
            }
            m_dGCache[av][a][o] = avNew;
            return avNew;
        }
        private AlphaVector alpha_b_a(BeliefState bs, Action a)
        {
            AlphaVector avSum = new AlphaVector(a);
            AlphaVector avGMax = null;
            double dValue = 0.0, dMaxValue = double.NegativeInfinity;
            foreach (Observation o in m_dDomain.Observations)
            {
                dMaxValue = double.NegativeInfinity;
                avGMax = null;
                foreach (AlphaVector avCurrent in m_lVectors)
                {
                    AlphaVector avG = alpha_a_o(a, o, avCurrent);
                    dValue = avG.InnerProduct(bs);
                    if (dValue > dMaxValue)
                    {
                        dMaxValue = dValue;
                        avGMax = avG;
                    }
                }
                avSum += avGMax;
            }
            avSum *= m_dDomain.DiscountFactor;
            AlphaVector avResult = new AlphaVector(a);
            foreach (State s in m_dDomain.States)
            {
                avResult[s] = avSum[s] + s.Reward(a);
            }
            return avResult;
        }

        private AlphaVector backup(BeliefState bs)
        {
            m_dGCache =  new Dictionary<AlphaVector, Dictionary<Action, Dictionary<Observation, AlphaVector>>>();
            AlphaVector avBest = new AlphaVector(), avCurrent = new AlphaVector();
            double dMaxValue = double.NegativeInfinity, dValue = 0.0;

            foreach (Action a in m_dDomain.Actions)
            {
                avCurrent = computeBestAlpha(a, bs); // alpha_a_b
                dValue = avCurrent.InnerProduct(bs);
                if (dValue > dMaxValue)
                {
                    avBest = avCurrent;
                    dMaxValue = dValue;
                }
            }
            return avBest;
        }

        private AlphaVector computeBestAlpha(Action action, BeliefState bs)
        {
            AlphaVector discountedRewardVector = new AlphaVector(action);

            foreach (Observation obs in m_dDomain.Observations)
            {
                AlphaVector cur_alpha_ao = null;
                AlphaVector best_alpha_ao = new AlphaVector();
                double best_val = double.NegativeInfinity;  double cur_val = 0;

                foreach (AlphaVector av in m_lVectors)
                {
                    cur_alpha_ao = computeAlphaAO(av, action, obs);
                    cur_val = cur_alpha_ao.InnerProduct(bs);
                    if (cur_val > best_val) {
                        best_alpha_ao = cur_alpha_ao;
                        best_val = cur_val;
                    }  
                }

                discountedRewardVector += best_alpha_ao;
            }

            discountedRewardVector = discountedRewardVector * m_dDomain.DiscountFactor;

            AlphaVector rA = new AlphaVector();
            foreach (State s in m_dDomain.States)
                rA[s] = s.Reward(action);

            return discountedRewardVector + rA;
        }
        
        private AlphaVector computeAlphaAO(AlphaVector alpha, Action a, Observation o)
        {
            AlphaVector res = new AlphaVector();
            foreach (State s in m_dDomain.States)
            {
                double accumulated_sum = 0;
                foreach (State succ in s.Successors(a))
                    accumulated_sum += (alpha[succ] * succ.ObservationProbability(a, o)) * s.TransitionProbability(a, succ);
                res[s] = accumulated_sum;
            }
            return res;
        }

        private HashSet<BeliefState> SimulateTrial(Policy p, int cMaxSteps)
        {
            BeliefState bsCurrent = m_dDomain.InitialBelief, bsNext = null;
            State sCurrent = bsCurrent.sampleState(), sNext = null; //Should be replaced with sampleState
            Action a = null;
            Observation o = null;
            HashSet<BeliefState> lBeliefs = new HashSet<BeliefState>();
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

        private HashSet<BeliefState> CollectBeliefs(int cBeliefs)
        {
            Debug.WriteLine("Started collecting " + cBeliefs + " points");
            RandomPolicy p = new RandomPolicy(m_dDomain);
            int cTrials = 100, cBeliefsPerTrial = cBeliefs / cTrials;
            HashSet<BeliefState> lBeliefs = new HashSet<BeliefState>();
            while (lBeliefs.Count < cBeliefs)
            {
                lBeliefs.UnionWith(SimulateTrial(p, cBeliefsPerTrial));
            }
            Debug.WriteLine("Collected " + lBeliefs.Count + " points");
            return lBeliefs;
        }

        private double ValueOf(BeliefState bs, List<AlphaVector> lVectors, out AlphaVector avBest)
        {
            double dValue = 0.0, dMaxValue = double.NegativeInfinity;
            avBest = null;
            foreach (AlphaVector av in lVectors)
            {
                dValue = av.InnerProduct(bs);
                if (dValue > dMaxValue)
                {
                    dMaxValue = dValue;
                    avBest = av;
                }
            }
            return dMaxValue;
        }

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

        public void PointBasedVI(int cBeliefs, int cMaxIterations)
        {

            //HashSet<BeliefState> beliefStates = generateInitialBS(cBeliefs); // B 
            HashSet<BeliefState> beliefStates = CollectBeliefs(cBeliefs);
            //m_dNextBeliefState = new Dictionary<KeyValuePair<BeliefState, KeyValuePair<Action, Observation>>, BeliefState>();
            /**foreach(BeliefState b in beliefStates)
            {
                foreach(Action a in m_dDomain.Actions)
                {
                    foreach(Observation o in m_dDomain.Observations)
                        m_dNextBeliefState.Add(new KeyValuePair<BeliefState, KeyValuePair<Action, Observation>>(b, new KeyValuePair<Action, Observation>(a, o)), b.Next(a, o));
                }
            }**/
            List<AlphaVector> vTag; //V' 

            m_lVectors = new List<AlphaVector>();
            m_lVectors.Add(createInitialAlphaVector());

            const double epsilon = 0.8;
            int iterationsLeft = cMaxIterations;

            while (iterationsLeft > 0)
            {
                vTag = new List<AlphaVector>();
                HashSet<BeliefState> beliefStatesLeftToImprove= new HashSet<BeliefState>(beliefStates); // B'

                while (beliefStatesLeftToImprove.Count() > 0)
                {
                    Console.WriteLine("Improvable belief states left");
                    Console.WriteLine(beliefStatesLeftToImprove.Count());

                    int ri = RandomGenerator.Next(beliefStatesLeftToImprove.Count());
                    HashSet<BeliefState>.Enumerator e = beliefStatesLeftToImprove.GetEnumerator();
                    for(int i=0; i<ri+1; i++)
                        e.MoveNext();
                    BeliefState sampledBS = e.Current;

                    Console.WriteLine("Iterations left: " + iterationsLeft);
                    Console.WriteLine("improvable bs left: " + beliefStatesLeftToImprove.Count());

                    AlphaVector alpha = backup(sampledBS);
                    AlphaVector alphaToAdd;

                    AlphaVector prevBestAlphaVector = null;
                    double prevValue = ValueOf(sampledBS, m_lVectors,out prevBestAlphaVector);
                    
                    if (alpha.InnerProduct(sampledBS) >= prevValue) // alpha is dominating, remove all belief states that are improved by it
                    {
                        Console.WriteLine("Found an improving vec");
                        HashSet<BeliefState> beliefStatesToRemove = new HashSet<BeliefState>();
                        foreach (BeliefState b_prime in beliefStatesLeftToImprove)
                        {
                            AlphaVector a = null;
                            if (alpha.InnerProduct(b_prime) >= ValueOf(b_prime, m_lVectors, out a)) // Keep only belief states which are not improved (prehaps left operand should be value for all V'?)
                                beliefStatesToRemove.Add(b_prime);
                        }
                        foreach (BeliefState b_to_remove in beliefStatesToRemove)
                            beliefStatesLeftToImprove.Remove(b_to_remove);

                        alphaToAdd = alpha;
                    }

                    else
                    {
                        beliefStatesLeftToImprove.Remove(sampledBS);
                        alphaToAdd = prevBestAlphaVector;
                    }

                    vTag.Add(alphaToAdd); //We either add the backedup alpha, or the best possible from V. Perhaps needs to be changed to SET
                }

                double diff = estimateDiff(m_lVectors, vTag, beliefStates);
                Console.WriteLine("Diff in current PERSUS iteration is:");

                Console.WriteLine(diff);

                if (diff < epsilon)
                    break;
                m_lVectors = vTag;
                iterationsLeft--;
            }
        }

        private double estimateDiff(List<AlphaVector> s1, List<AlphaVector> s2, HashSet<BeliefState> beliefStates)
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

        private HashSet<BeliefState> generateInitialBS(int numToGenerate)
        {
            HashSet<BeliefState> res = new HashSet<BeliefState>();
            BeliefState curBeliefState = m_dDomain.InitialBelief;
            Action[] actionsArray = m_dDomain.Actions.Cast<Action>().ToArray();
            int numOfActions = actionsArray.Count();

            while (res.Count() < numToGenerate)
            {
                int actionIndex = RandomGenerator.Next(numOfActions);
                Action curAction = actionsArray[actionIndex];
                Observation curObs = curBeliefState.sampleState().Apply(curAction).RandomObservation(curAction);
                curBeliefState = curBeliefState.Next(curAction, curObs);
                res.Add(curBeliefState);
            }
            return res;
        }
    }
}
