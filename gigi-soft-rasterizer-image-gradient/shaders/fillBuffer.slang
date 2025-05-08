// Unnamed technique, shader fillBuffer
/*$(ShaderResources)*/

/*$(_compute:fillBuffer)*/(uint3 DTid : SV_DispatchThreadID)
{
    // if (/*$(Variable:beginDescent)*/)
    //     return;

    // Populate the vertices buffer from Gigi variables
    vertices[0] = /*$(Variable:v0)*/;
    vertices[1] = /*$(Variable:v1)*/;
    vertices[2] = /*$(Variable:v2)*/;

    // Populate the color buffer from the Gigi color variable
    color[0] = /*$(Variable:color)*/;

}

/*
Shader Resources:
	Buffer vertices (as UAV)
*/
