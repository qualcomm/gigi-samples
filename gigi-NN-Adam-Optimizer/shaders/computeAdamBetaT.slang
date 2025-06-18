/*$(ShaderResources)*/

/*$(_compute:ComputeAdamBeta)*/(uint3 DTid : SV_DispatchThreadID)
{

    float adamBeta1 = /*$(Variable:adamBeta1)*/ ;
    float adamBeta2 = /*$(Variable:adamBeta2)*/;

    uint trainingSteps = /*$(Variable:frameNumber)*/;

    nnData[0].adamBeta1T = pow(adamBeta1, trainingSteps + 1);
    nnData[0].adamBeta2T = pow(adamBeta2, trainingSteps + 1);
}
