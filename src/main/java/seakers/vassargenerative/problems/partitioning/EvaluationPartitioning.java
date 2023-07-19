package seakers.vassargenerative.problems.partitioning;

import org.moeaframework.algorithm.EpsilonMOEA;
import org.moeaframework.core.*;
import org.moeaframework.core.comparator.ChainedComparator;
import org.moeaframework.core.comparator.ParetoObjectiveComparator;
import org.moeaframework.core.operator.*;
import org.moeaframework.core.operator.binary.BitFlip;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.util.TypedProperties;
import seakers.architecture.util.IntegerVariable;
import seakers.vassarexecheur.search.intialization.SynchronizedMersenneTwister;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningArchitecture;
import seakers.vassarexecheur.search.problems.partitioning.PartitioningProblem;
import seakers.vassarheur.BaseParams;
import seakers.vassarheur.evaluation.AbstractArchitectureEvaluator;
import seakers.vassarheur.evaluation.ArchitectureEvaluationManager;
import seakers.vassarheur.problems.PartitioningAndAssigning.ArchitectureEvaluator;
import seakers.vassarheur.problems.PartitioningAndAssigning.ClimateCentricPartitioningParams;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
public class EvaluationPartitioning {

    public static void main(String[] args) {
       int[] instrumentArray = {0,0,1,2,2,2,3,3,4,5,6,7};
       int[] orbitArray = {2,1,2,4,0,3,1,2};
        ArrayList<Integer> instruments = new ArrayList<Integer>();
        ArrayList<Integer> orbits = new ArrayList<Integer>();
        for(int i=0;i<instrumentArray.length-1;i++){
            instruments.add(instrumentArray[i]);

        }

        for(int i=0;i<orbitArray.length-1;i++){
            orbits.add(orbitArray[i]);

        }

       double[] objectives = EvaluatePythonArchitecture(instruments,orbits);
        System.out.println("Science: " + Double.toString(objectives[0]));
        System.out.println("Cost: " + Double.toString(objectives[1]));

}


    public static double[] EvaluatePythonArchitecture(ArrayList<Integer> instruments, ArrayList<Integer> orbits) {
        int numCpus = 1;

        ExecutorService pool = Executors.newFixedThreadPool(numCpus);
        // Get time
        String timestamp = new SimpleDateFormat("yyyy-MM-dd-HH-mm").format(new Date());

        // Heuristic Enforcement Methods
        /**
         * dutyCycleConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         * instrumentOrbitRelationsConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         * interferenceConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         * packingEfficiencyConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         * spacecraftMassConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         * synergyConstrained = [interior_penalty, AOS, biased_init, ACH, objective, constraint]
         *
         * heuristicsConstrained = [dutyCycleConstrained, instrumentOrbitRelationsConstrained, interferenceConstrained, packingEfficiencyConstrained, spacecraftMassConstrained, synergyConstrained]
         */
        boolean[] dutyCycleConstrained = {false, false, false, false, false, false};
        boolean[] instrumentOrbitRelationsConstrained = {false, false, false, false, false, false};
        boolean[] interferenceConstrained = {false, false, false, false, false, false};
        boolean[] packingEfficiencyConstrained = {false, false, false, false, false, false};
        boolean[] spacecraftMassConstrained = {false, false, false, false, false, false};
        boolean[] synergyConstrained = {false, false, false, false, false, false};

        boolean[][] heuristicsConstrained = new boolean[6][6];
        for (int i = 0; i < 6; i++) {
            heuristicsConstrained[0][i] = dutyCycleConstrained[i];
            heuristicsConstrained[1][i] = instrumentOrbitRelationsConstrained[i];
            heuristicsConstrained[2][i] = interferenceConstrained[i];
            heuristicsConstrained[3][i] = packingEfficiencyConstrained[i];
            heuristicsConstrained[4][i] = spacecraftMassConstrained[i];
            heuristicsConstrained[5][i] = synergyConstrained[i];
        }

        int numberOfHeuristicConstraints = 0;
        int numberOfHeuristicObjectives = 0;
        for (int i = 0; i < 6; i++) {
            if (heuristicsConstrained[i][5]) {
                numberOfHeuristicConstraints++;
            }
            if (heuristicsConstrained[i][4]) {
                numberOfHeuristicObjectives++;
            }
        }


        // Set seed for random number generator
        //PRNG.setSeed(4321);

        //setup for epsilon MOEA
        double dcThreshold = 0.5;
        double massThreshold = 3000.0; // [kg]
        double packEffThreshold = 0.7;
        boolean considerFeasibility = true; // use false only for biased random generation for random population runs

        String savePath = System.getProperty("user.dir") + File.separator + "results";

        String resourcesPath = "C:\\Users\\dforn\\Documents\\TEXASAM\\PROJECTS\\VASSAR_resources"; // for lab system
        //String resourcesPath = "C:\\Users\\rosha\\Documents\\SEAK Lab Github\\VASSAR\\VASSAR_resources-heur"; // for laptop

        ClimateCentricPartitioningParams params = new ClimateCentricPartitioningParams(resourcesPath, "FUZZY-ATTRIBUTES", "test", "normal");

        HashMap<String, String[]> instrumentSynergyMap = getInstrumentSynergyNameMap(params);
        HashMap<String, String[]> interferingInstrumentsMap = getInstrumentInterferenceNameMap(params);

        AbstractArchitectureEvaluator evaluator = new ArchitectureEvaluator(considerFeasibility, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold);
        ArchitectureEvaluationManager evaluationManager = new ArchitectureEvaluationManager(params, evaluator);
        evaluationManager.init(numCpus);

        PRNG.setRandom(new SynchronizedMersenneTwister());
        PartitioningProblem problem = new PartitioningProblem(params.getProblemName(), evaluationManager, params, interferingInstrumentsMap, instrumentSynergyMap, dcThreshold, massThreshold, packEffThreshold, numberOfHeuristicObjectives, numberOfHeuristicConstraints, heuristicsConstrained);



        System.out.println("Evaluating the generated architecture for the Partitioning Problem");
        int[] instrumentPartitioning = new int[params.getNumInstr()];
        int[] orbitAssignment = new int[params.getNumInstr()];
        int numints = params.getNumInstr();
        // There must be at least one satellite
        int maxNumSats = PRNG.nextInt(params.getNumInstr()) + 1;

        for(int j = 0; j < params.getNumInstr(); j++){
            instrumentPartitioning[j] = instruments.get(j);
            orbitAssignment[j] = orbits.get(j);

        }


        HashMap<Integer, Integer> map = new HashMap<>();
        int satIndex = 0;
        /*for(int m = 0; m < params.getNumInstr(); m++){
            int satID = instrumentPartitioning[m];
            if(map.keySet().contains(satID)){
                instrumentPartitioning[m] = map.get(satID);
            }else{
                instrumentPartitioning[m] = satIndex;
                map.put(satID, satIndex);
                satIndex++;
            }
        }*/
        //Arrays.sort(instrumentPartitioning);

        //int numSats = map.keySet().size();
        /*for(int n = 0; n < params.getNumInstr(); n++){
            if(n < numSats){
                orbitAssignment[n] = orbits.get(n);
            }else{
                orbitAssignment[n] = -1;
            }
        }*/

        PartitioningArchitecture arch = createPartitioningArchitecture(instrumentPartitioning, orbitAssignment, params);

        problem.evaluateArch(arch);

        pool.shutdown();
        evaluationManager.clear();
        System.out.println("DONE");


        double[] objectives = arch.getObjectives();
        if (objectives[1]>=1)
            objectives[1]=1;



        return objectives;
    }



    protected static HashMap<String, String[]> getInstrumentSynergyNameMap(BaseParams params) {
        HashMap<String, String[]> synergyNameMap = new HashMap<>();
        if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
            synergyNameMap.put("ACE_ORCA", new String[]{"DESD_LID", "GACM_VIS", "ACE_POL", "HYSP_TIR", "ACE_LID"});
            synergyNameMap.put("DESD_LID", new String[]{"ACE_ORCA", "ACE_LID", "ACE_POL"});
            synergyNameMap.put("GACM_VIS", new String[]{"ACE_ORCA", "ACE_LID"});
            synergyNameMap.put("HYSP_TIR", new String[]{"ACE_ORCA", "POSTEPS_IRS"});
            synergyNameMap.put("ACE_POL", new String[]{"ACE_ORCA", "DESD_LID"});
            synergyNameMap.put("ACE_LID", new String[]{"ACE_ORCA", "CNES_KaRIN", "DESD_LID", "GACM_VIS"});
            synergyNameMap.put("POSTEPS_IRS", new String[]{"HYSP_TIR"});
            synergyNameMap.put("CNES_KaRIN", new String[]{"ACE_LID"});
        }
        else {
            System.out.println("Synergy Map for current problem not formulated");
        }
        return synergyNameMap;
    }

    /**
     * Creates instrument interference map used to compute the instrument interference violation heuristic (only formulated for the
     * Climate Centric problem for now)
     * @param params
     * @return Instrument interference hashmap
     */





        protected static HashMap<String, String[]> getInstrumentInterferenceNameMap(BaseParams params) {
        HashMap<String, String[]> interferenceNameMap = new HashMap<>();
        if (params.getProblemName().equalsIgnoreCase("ClimateCentric")) {
            interferenceNameMap.put("ACE_LID", new String[]{"ACE_CPR", "DESD_SAR", "CLAR_ERB", "GACM_SWIR"});
            interferenceNameMap.put("ACE_CPR", new String[]{"ACE_LID", "DESD_SAR", "CNES_KaRIN", "CLAR_ERB", "ACE_POL", "ACE_ORCA", "GACM_SWIR"});
            interferenceNameMap.put("DESD_SAR", new String[]{"ACE_LID", "ACE_CPR"});
            interferenceNameMap.put("CLAR_ERB", new String[]{"ACE_LID", "ACE_CPR"});
            interferenceNameMap.put("CNES_KaRIN", new String[]{"ACE_CPR"});
            interferenceNameMap.put("ACE_POL", new String[]{"ACE_CPR"});
            interferenceNameMap.put("ACE_ORCA", new String[]{"ACE_CPR"});
            interferenceNameMap.put("GACM_SWIR", new String[]{"ACE_LID", "ACE_CPR"});
        }
        else {
            System.out.println("Interference Map fpr current problem not formulated");
        }
        return interferenceNameMap;
    }

    public static PartitioningArchitecture createPartitioningArchitecture (int[] instrumentPartitions, int[] orbitAssignments, ClimateCentricPartitioningParams params) {
        PartitioningArchitecture arch = new PartitioningArchitecture(params.getNumInstr(), params.getNumOrbits(), 2, params);

        for (int p = 0; p < params.getNumInstr(); p++) {
            IntegerVariable var = new IntegerVariable(instrumentPartitions[p], 0, params.getNumInstr());
            arch.setVariable(p, var);
        }

        for (int q = 0; q < params.getNumInstr(); q++) {
            IntegerVariable var = new IntegerVariable(orbitAssignments[q], -1, params.getNumOrbits());
            arch.setVariable(params.getNumInstr() + q, var);
        }
        return arch;
    }


}
