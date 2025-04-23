from phe import paillier
#generate public and private key pairs.
public_key, private_key = paillier.generate_paillier_keypair()

a = 1
b = 2

#Task 1
enc_a = public_key.encrypt(a)
enc_b = public_key.encrypt(b)

print(enc_a)
print(enc_b)

enc_c = enc_a + enc_b
print(enc_c._EncryptedNumber__ciphertext)

#Task 2
enc_d = enc_c * 2
print(enc_d._EncryptedNumber__ciphertext)

#Task 3
private_key.decrypt(enc_d)

#Question 2
er = enc_c*enc_c 
print(er)