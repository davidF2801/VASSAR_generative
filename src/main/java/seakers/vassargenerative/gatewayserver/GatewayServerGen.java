package seakers.vassargenerative.gatewayserver;
import py4j.GatewayServer;
import seakers.vassargenerative.problems.assigning.EvaluationAssigning;
import seakers.vassargenerative.problems.partitioning.EvaluationPartitioning;


public class GatewayServerGen {

    public static void main(String[] args) {
        //EvaluationAssigning myJavaObject = new EvaluationAssigning();
        EvaluationPartitioning myJavaObject = new EvaluationPartitioning();
        py4j.GatewayServer gatewayServer = new py4j.GatewayServer(myJavaObject);
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}



 