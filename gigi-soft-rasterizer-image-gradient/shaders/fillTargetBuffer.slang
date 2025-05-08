// Unnamed technique, shader fillBuffer
/*$(ShaderResources)*/

/*$(_compute:fillBuffer)*/(uint3 DTid : SV_DispatchThreadID)
{
    vertices[0] = /*$(Variable:target_v0)*/;
    vertices[1] = /*$(Variable:target_v1)*/;
    vertices[2] = /*$(Variable:target_v2)*/;
}

/*
Shader Resources:
	Buffer vertices (as UAV)
*/
