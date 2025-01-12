from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import UserInput
from django.contrib.auth import authenticate, login, logout
from .serializers import UserInputSerializer, CustomUserSerializer
from rest_framework.permissions import AllowAny
from rest_framework import generics, status
from myproject.code_check import security_check
from myproject.AST_tree_creation import Generate, test_and_combine

@api_view(['POST'])
def reverse_text(request):
    """
    This view takes text from the request, processes it to generate a class-like structure,
    and returns the generated class.
    """
    try:
        text = request.data.get('text', '')
        generated_class = Generate(text)  # Generate class from text
        return Response({'reversed_text': generated_class}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def test(request):
    """
    This view takes text (presumably modified class from previous endpoint),
    tests it, and combines it with other elements if necessary, returning the final combined output.
    """
    try:
        text = request.data.get('text', '')
        combined_output = test_and_combine(text)  # Process and combine the text
        return Response({'combined_text': combined_output}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    


@api_view(['POST'])
def add_is_cool(request):
    text = request.data.get('text')
    text_with_is_cool = security_check(text)
    return Response({'text_with_is_cool': text_with_is_cool})


class UserRegisterView(APIView):
    def post(self, request):
        serializer = CustomUserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserLoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            return Response({'message': 'Login successful.'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Invalid credentials.'}, status=status.HTTP_401_UNAUTHORIZED)

class UserLogoutView(APIView):
    def get(self, request, *args, **kwargs):
        logout(request)
        return Response({'message': 'Logout successful.'}, status=status.HTTP_200_OK)