package seakers.vassargenerative.gatewayserver;
import py4j.GatewayServer;
import seakers.vassargenerative.problems.assigning.EvaluationAssigning;

public class GatewayServerGen {

    public static void main(String[] args) {
        EvaluationAssigning myJavaObject = new EvaluationAssigning();
        py4j.GatewayServer gatewayServer = new py4j.GatewayServer(myJavaObject);
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }
}



 