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


        private AlphaVector G(Action a, Observation o, AlphaVector av)
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
        private AlphaVector G(BeliefState bs, Action a)
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
                    AlphaVector avG = G(a, o, avCurrent);
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
            AlphaVector discounted_sum = new AlphaVector(action);

            foreach (Observation obs in m_dDomain.Observations)
            {
                AlphaVector cur_alpha_ao = null;
                //AlphaVector best_alpha_ao = defaultVector;
                AlphaVector best_alpha_ao = new AlphaVector();
                double best_val = double.NegativeInfinity;
                double cur_val = 0;

                foreach (AlphaVector av in m_lVectors)
                {
                    cur_alpha_ao = computeAlphaAO(av, action, obs);
                    cur_val = cur_alpha_ao.InnerProduct(bs);
                    if (cur_val > best_val) {
                        best_alpha_ao = cur_alpha_ao;
                        best_val = cur_val;
                    }  
                }
                discounted_sum += best_alpha_ao;
            }
            discounted_sum *= m_dDomain.DiscountFactor;

            AlphaVector rA = new AlphaVector();
            foreach (State s in m_dDomain.States)
                rA[s] = s.Reward(action);

            return discounted_sum + rA;
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

        private List<BeliefState> SimulateTrial(Policy p, int cMaxSteps)
        {
            BeliefState bsCurrent = m_dDomain.InitialBelief, bsNext = null;
            State sCurrent = bsCurrent.sampleState(), sNext = null; //Should be replaced with sampleState
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

        public void PointBasedVI(int cBeliefs, int cMaxIterations)
        {

            HashSet<BeliefState> BeliefStates = generateInitialBS(cBeliefs); // B

            ////PUT IN DIFFERENT FUNCTION THE V0 INIT
            AlphaVector V0 = new AlphaVector();
            double minReward = Double.PositiveInfinity;
            foreach(State s in m_dDomain.States)
            {
                foreach(Action a in m_dDomain.Actions)
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

            m_lVectors = new List<AlphaVector>();
            m_lVectors.Add(V0);

            const double epsilon = 0.1;
            List<AlphaVector> curAlphaSet; //V'
            int iterationsLeft = cMaxIterations;

            while (iterationsLeft > 0)
            {
                curAlphaSet = new List<AlphaVector>();
                HashSet<BeliefState> improvableBeliefStates= new HashSet<BeliefState>(BeliefStates); // B'

                while (improvableBeliefStates.Count() > 0)
                {
                    Console.WriteLine("Improvable belief states left");
                    Console.WriteLine(improvableBeliefStates.Count());

                    //int ri = RandomGenerator.Next(improvableBeliefStates.Count());
                    HashSet<BeliefState>.Enumerator e = improvableBeliefStates.GetEnumerator();
                    e.MoveNext();
                    BeliefState B = e.Current;

                    AlphaVector alpha = backup(B);
                    AlphaVector alpha_to_add;

                    AlphaVector prev_alpha_argmax = null;
                    double prevValue = ValueOf(B, m_lVectors,out prev_alpha_argmax);
                    
                    if (alpha.InnerProduct(B) >= prevValue) // alpha is dominating, remove all belief states that are improved
                    {
                        HashSet<BeliefState> beliefStatesToRemove = new HashSet<BeliefState>();
                        foreach (BeliefState b_prime in improvableBeliefStates)
                        {
                            if (alpha.InnerProduct(b_prime) >= prevValue) // Keep only belief states which are not improved (prehaps left operand should be value for all V'?)
                                beliefStatesToRemove.Add(b_prime);
                        }
                        foreach (BeliefState b_to_remove in beliefStatesToRemove)
                            improvableBeliefStates.Remove(b_to_remove);

                        alpha_to_add = alpha;
                    }
                    else
                    {
                        improvableBeliefStates.Remove(B);
                        alpha_to_add = prev_alpha_argmax;
                    }
                    curAlphaSet.Add(alpha_to_add); //We either add the backedup alpha, or the best possible from V. Perhaps needs to be changed to SET
                }

                double diff = differenceValue(m_lVectors, curAlphaSet, BeliefStates);
                Console.WriteLine("Diff in current PERSUS iteration is:");

                Console.WriteLine(diff);

                if ( diff < epsilon)
                    break;
                m_lVectors = curAlphaSet;
                iterationsLeft--;
            }
        }

        private double differenceValue(IEnumerable<AlphaVector> s1, IEnumerable<AlphaVector> s2, IEnumerable<BeliefState> beliefStates)
        {
            double maxDiff = Double.NegativeInfinity;
            foreach(BeliefState b in beliefStates)
            {
                double curDiff = Math.Abs(calculateMaxAlphaVector(b, s1).InnerProduct(b) - calculateMaxAlphaVector(b, s2).InnerProduct(b)); 
                if (curDiff > maxDiff)
                    maxDiff = curDiff;
            } 
            return maxDiff;
        }

        private AlphaVector calculateMaxAlphaVector(BeliefState beliefState, IEnumerable<AlphaVector> alphaVectors) // consider change to ValueOf
        {
            double maxScore = 0;
            AlphaVector res = new AlphaVector();
            foreach (AlphaVector alpha in alphaVectors)
            {
                double score = alpha.InnerProduct(beliefState);
                if (score > maxScore)
                {
                    maxScore = score;
                    res = alpha;
                }
            }
            return res;
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
